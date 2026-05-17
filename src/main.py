import torch
from torch.utils.data import DataLoader
import sys
from tqdm import tqdm
import numpy as np
import os
import argparse

from collections import defaultdict
from dumpManager.RamDumpDataset import RamDumpDataset
from transformers.bytesClassifier.BytesTransformerClassifier import BytesTransformerClassifier
import config as cfg

from tools.visualizerExport import RaidVisualizerExporter
from utils.checkpointing import resolve_checkpoint_path, _unwrap_compiled_state_dict

def evaluate(genereateExport=False, checkpoint_name=None):
    if not torch.cuda.is_available():
        print("Aucun GPU détécté, évaluation sur CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        print(f"--- Évaluation sur {device} ---")

    print("Construction du modèle...")
    model_path = resolve_checkpoint_path(checkpoint_name, is_model=True)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model_config = checkpoint.get("model_config", cfg.MODEL_CONFIG) if isinstance(checkpoint, dict) else cfg.MODEL_CONFIG
        model = BytesTransformerClassifier(**model_config)
        model.to(device)

        useBf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
        print(f"bf16 activé: {useBf16}")

        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        # Unwrap compiled model state_dict if necessary
        state_dict = _unwrap_compiled_state_dict(state_dict)
        try:
            load_result = model.load_state_dict(state_dict)
        except RuntimeError as load_error:
            print(f"Avertissement: chargement strict impossible ({load_error}). Repli sur strict=False.")
            load_result = model.load_state_dict(state_dict, strict=False)

        if getattr(load_result, "missing_keys", None):
            print(f"Clés manquantes lors du chargement: {load_result.missing_keys}")
        if getattr(load_result, "unexpected_keys", None):
            print(f"Clés inattendues lors du chargement: {load_result.unexpected_keys}")
        print(f"Poids chargés avec succès depuis : {model_path}")
    except FileNotFoundError:
        print(f"ERREUR CRITIQUE : Le fichier modèle '{model_path}' est introuvable.")
        sys.exit(1)
    except Exception as e:
        print(f"ERREUR lors du chargement des poids : {e}")
        sys.exit(1)

    model.eval()

    print("Chargement du Dataset...")
    test_dataset = RamDumpDataset(
        bin_path=cfg.BIN_PATH,
        meta_path=cfg.META_PATH,
        chunk_size=cfg.EVAL_DATASET_CONFIG["chunk_size"],
        offset=cfg.EVAL_DATASET_CONFIG["offset"]
    )

    # Creates buffers for metadata corresponding to each byte offset
    meta_starts = [entry['ds'] for entry in test_dataset.metadata]
    meta_starts_np = np.asarray(meta_starts, dtype=np.int64)
    meta_ds_np = np.asarray([entry['ds'] for entry in test_dataset.metadata], dtype=np.int64)
    meta_de_np = np.asarray([entry['de'] for entry in test_dataset.metadata], dtype=np.int64)
    meta_types_np = np.asarray([entry['t'] for entry in test_dataset.metadata], dtype=object)

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.EVAL_CONFIG["batch_size"],
        shuffle=False,
        num_workers=cfg.EVAL_LOADER_CONFIG["num_workers"],
        pin_memory=cfg.EVAL_LOADER_CONFIG["pin_memory"],
        prefetch_factor=cfg.EVAL_LOADER_CONFIG["prefetch_factor"],
        persistent_workers=True,
    )

    visualizer = RaidVisualizerExporter() if genereateExport else None

    print("Démarrage de l'analyse...")
    total, correct = 0, 0
    errorType = defaultdict(int)

    # Global confidence-weighted vote buffers indexed by absolute byte offset.
    # This keeps context stable even when windows cross batch boundaries.
    bin_size = int(meta_de_np[-1]) if len(meta_de_np) > 0 else 0
    vote_sum = np.zeros(bin_size, dtype=np.float32)
    vote_weight = np.zeros(bin_size, dtype=np.float32)
    label_sum = np.zeros(bin_size, dtype=np.float32)
    label_weight = np.zeros(bin_size, dtype=np.float32)

    with torch.no_grad():

        for _, (data, labels, start_offsets) in enumerate(tqdm(test_loader, desc="Inférence R.A.I.D.", unit="batch")):
            data, labels = data.to(device), labels.to(device)

            if useBf16:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                    logits = model(data)
            else:
                logits = model(data)

            # Convert logits into probs
            probs = torch.sigmoid(logits) # 32*512 | batch_size * sequence_length -> Tensor
            predictions = (probs > 0.5).float() # batch_size * sequence_length
            confidence = torch.abs(probs - 0.5) * 2 # Converts ranges [0.5, 1] to [0, 1] for crypted and [0, 0.5] to [1, 0] for clear

            # Only evaluate on labeled positions (ignore label == -1 for padding)
            valid_labels = labels >= 0
            if valid_labels.any():
                total += valid_labels.sum().item()
                correct += ((predictions[valid_labels] == labels[valid_labels].float()).sum().item())

            # Build absolute byte indices for each prediction in the batch.
            batch_offsets = start_offsets.cpu().numpy().astype(np.int64)
            batch_probs = probs.cpu().float().numpy().astype(np.float32)
            batch_labels = labels.cpu().float().numpy().astype(np.float32)
            batch_conf = confidence.cpu().float().numpy().astype(np.float32)
            sample_len = batch_probs.shape[1]
            local_positions = np.arange(sample_len, dtype=np.int64)
            abs_offsets = batch_offsets[:, None] + local_positions[None, :]

            # Safety mask for the last, possibly incomplete, coverage area.
            in_range = (abs_offsets >= 0) & (abs_offsets < bin_size)
            # Also mask out unlabeled positions (label == -1)
            valid_mask = batch_labels >= 0
            final_mask = in_range & valid_mask
            if not np.any(final_mask):
                continue

            # Aggregate votes and labels into global buffers using advanced indexing with np.add.at to handle duplicates.
            flat_offsets = abs_offsets[final_mask]
            flat_probs = batch_probs[final_mask]
            flat_labels = batch_labels[final_mask]
            flat_weights = np.clip(batch_conf[final_mask], 1e-6, 1.0)

            np.add.at(vote_sum, flat_offsets, flat_probs * flat_weights)
            np.add.at(vote_weight, flat_offsets, flat_weights)
            np.add.at(label_sum, flat_offsets, flat_labels)
            np.add.at(label_weight, flat_offsets, np.ones_like(flat_labels))

    # Final per-byte aggregated prediction using confidence-weighted vote.
    # Covered mask is representing the offsets that received at least one vote, allowing us to distinguish between "unknown" (no votes) and "clear/crypted" (with votes).
    covered_mask = vote_weight > 0
    agg_prob = np.zeros_like(vote_sum)
    agg_prob[covered_mask] = vote_sum[covered_mask] / vote_weight[covered_mask]
    agg_pred = np.zeros_like(vote_sum, dtype=np.int8)
    agg_pred[covered_mask] = (agg_prob[covered_mask] > 0.5).astype(np.int8)

    # Aggregate labels using the weight accumulated during voting
    # This allows us to compute an average label for each offset, which is then rounded to get the final aggregated label.
    covered_offsets = np.flatnonzero(covered_mask)
    label_covered = label_weight > 0
    agg_label = np.zeros_like(label_sum, dtype=np.int8)
    agg_label[label_covered] = np.round(label_sum[label_covered] / label_weight[label_covered]).astype(np.int8)

    # Confidence of the aggregated prediction is derived from the average confidence of contributing predictions.
    # We convert it to a [0, 1] range where 1 means "all votes agree and are confident" and 0 means "all votes are around the 0.5 threshold".
    agg_conf = np.zeros_like(agg_prob)
    agg_conf[covered_mask] = np.abs(agg_prob[covered_mask] - 0.5) * 2.0

    # If there are covered offsets, perform error analysis and prepare visualization segments.
    if np.any(covered_mask):

        pred_seq = agg_pred[covered_offsets]
        label_seq = agg_label[covered_offsets]
        correct_seq = pred_seq == label_seq
        conf_seq = agg_conf[covered_offsets]

        # Run segmentation on global offsets, splitting when coverage is not contiguous
        # or when prediction/correctness flips.
        run_breaks = np.zeros(covered_offsets.shape[0], dtype=bool)
        run_breaks[0] = True
        if covered_offsets.shape[0] > 1:
            run_breaks[1:] = (
                (np.diff(covered_offsets) != 1)
                | (np.diff(pred_seq) != 0)
                | (np.diff(correct_seq.astype(np.int8)) != 0)
            )

        run_starts = np.flatnonzero(run_breaks)
        run_ends = np.concatenate((run_starts[1:], [covered_offsets.shape[0]]))

        byte_offsets = covered_offsets[run_starts]
        run_preds = pred_seq[run_starts]
        run_correct = correct_seq[run_starts]
        segment_sizes = covered_offsets[run_ends - 1] - covered_offsets[run_starts] + 1
        run_confidences = np.fromiter(
            (float(conf_seq[s:e].mean()) for s, e in zip(run_starts, run_ends)),
            dtype=np.float32,
            count=run_starts.shape[0],
        )

        # Determine the real type for each run based on the run start byte offset.
        idxs = np.searchsorted(meta_starts_np, byte_offsets, side="right") - 1
        real_types = np.full(byte_offsets.shape, "unknown", dtype=object)
        valid_idx_mask = idxs >= 0
        if np.any(valid_idx_mask):
            valid_positions = np.flatnonzero(valid_idx_mask)
            valid_meta_idx = idxs[valid_idx_mask]
            valid_offsets = byte_offsets[valid_idx_mask]
            in_range_mask = (meta_ds_np[valid_meta_idx] <= valid_offsets) & (valid_offsets < meta_de_np[valid_meta_idx])

            if np.any(in_range_mask):
                in_range_positions = valid_positions[in_range_mask]
                in_range_meta_idx = valid_meta_idx[in_range_mask]
                real_types[in_range_positions] = meta_types_np[in_range_meta_idx]

            out_of_range_count = int(np.count_nonzero(~in_range_mask))
            if out_of_range_count > 0:
                print(f"[ERROR] {out_of_range_count} offsets tombent dans un gap de metadata après agrégation.")

        # Some error handling for invalid global offsets
        typeTotal = defaultdict(int)
        unique_types_all, inverse_idx_all = np.unique(real_types, return_inverse=True)
        weighted_totals = np.bincount(inverse_idx_all, weights=segment_sizes)
        for t, total_size in zip(unique_types_all, weighted_totals):
            typeTotal[t] += int(total_size)

        error_mask = ~run_correct
        if np.any(error_mask):
            err_types = real_types[error_mask]
            err_sizes = segment_sizes[error_mask]
            unique_types, inverse_idx = np.unique(err_types, return_inverse=True)
            weighted_errors = np.bincount(inverse_idx, weights=err_sizes)
            for err_type, err_count in zip(unique_types, weighted_errors):
                errorType[err_type] += int(err_count)

        # Théo's visualiser shit, honestly, it does not need any comments
        if visualizer:
            for byte_offset, segment_size, run_pred, is_correct, real_type, run_conf in zip(
                byte_offsets,
                segment_sizes,
                run_preds,
                run_correct,
                real_types,
                run_confidences,
            ):
                visualizer.addSegment(
                    offset=int(byte_offset),
                    size=int(segment_size),
                    prediction="crypted" if run_pred == 1 else "clear",
                    isCorrect=bool(is_correct),
                    trueLabel=real_type,
                    metadata={"confidence": float(run_conf)},
                )

    accuracy = correct / total if total > 0 else 0
    aggregated_total = int(np.count_nonzero(covered_mask))
    aggregated_correct = int(np.count_nonzero(agg_pred[covered_mask] == agg_label[covered_mask])) if aggregated_total > 0 else 0
    aggregated_accuracy = (aggregated_correct / aggregated_total) if aggregated_total > 0 else 0
    print("\n--- Détails des erreurs par type de données ---")
    total_errors = sum(errorType.values())
    for data_type, count in sorted(errorType.items(), key=lambda x: -x[1]):
        proportion_global = (count / total_errors * 100) if total_errors > 0 else 0
        type_total = typeTotal.get(data_type, 0)
        proportion_type = (count / type_total * 100) if type_total > 0 else 0
        print(
            f"  Type: {data_type:<15} | Erreurs: {count:<8} "
            f"| Prop. globale: {proportion_global:>6.2f}% "
            f"| Erreurs/Type: {proportion_type:>6.2f}%"
        )
    print(f"\n--- Évaluation terminée ---")
    print(f"Exactitude brute (avant vote): {accuracy:.2%} ({correct}/{total})")
    print(f"Exactitude agrégée (vote pondéré): {aggregated_accuracy:.2%} ({aggregated_correct}/{aggregated_total})")

    if visualizer:
        visualizer.saveJson(filepath=f"{cfg.VISUAL_EXPORT_DIR}/raid_evaluation_visualization.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évalue le modèle R.A.I.D sur un dump RAM.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Nom de checkpoint (dans checkpoints/), chemin relatif ou absolu. "
            "Si absent, utilise MODEL_PATH depuis config.py."
        ),
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Désactive la génération du fichier d'export visualizer.",
    )
    args = parser.parse_args()

    evaluate(genereateExport=not args.no_export, checkpoint_name=args.checkpoint)