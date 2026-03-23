import torch
from torch.utils.data import DataLoader
import sys
import numpy as np

from collections import defaultdict
from dumpManager.RamDumpDataset import RamDumpDataset
from transformers.bytesClassifier.BytesTransformerClassifier import BytesTransformerClassifier
import config as cfg

from tools.visualizerExport import RaidVisualizerExporter

def evaluate(genereateExport=False):
    if not torch.cuda.is_available():
        print("Aucun GPU détécté, évaluation sur CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        print(f"--- Évaluation sur {device} ---")

    print("Construction du modèle...")
    model = BytesTransformerClassifier(
        dim_model=128, num_heads=4, num_layers=2
    )
    model.to(device)

    try:
        model.load_state_dict(torch.load(cfg.MODEL_PATH, map_location=device))
        print(f"Poids chargés avec succès depuis : {cfg.MODEL_PATH}")
    except FileNotFoundError:
        print(f"ERREUR CRITIQUE : Le fichier modèle '{cfg.MODEL_PATH}' est introuvable.")
        sys.exit(1)
    except Exception as e:
        print(f"ERREUR lors du chargement des poids : {e}")
        sys.exit(1)

    model.eval()

    print("Chargement du Dataset...")
    test_dataset = RamDumpDataset(
        bin_path=cfg.BIN_PATH,
        meta_path=cfg.META_PATH,
        chunk_size=512,
        offset=128
    )

    # Creates buffers for metadata corresponding to each byte offset
    meta_starts = [entry['ds'] for entry in test_dataset.metadata]
    meta_starts_np = np.asarray(meta_starts, dtype=np.int64)
    meta_ds_np = np.asarray([entry['ds'] for entry in test_dataset.metadata], dtype=np.int64)
    meta_de_np = np.asarray([entry['de'] for entry in test_dataset.metadata], dtype=np.int64)
    meta_types_np = np.asarray([entry['t'] for entry in test_dataset.metadata], dtype=object)

    BATCH_SIZE = 32
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

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

        for _, (data, labels, start_offsets) in enumerate(test_loader):
            data, labels = data.to(device), labels.to(device)

            logits = model(data)

            # Convert logits into probs
            probs = torch.sigmoid(logits) # 32*512 | batch_size * sequence_length -> Tensor
            predictions = (probs > 0.5).float() # batch_size * sequence_length
            confidence = torch.abs(probs - 0.5) * 2 # Converts ranges [0.5, 1] to [0, 1] for crypted and [0, 0.5] to [1, 0] for clear

            # Global accuracy at raw prediction level (before vote aggregation).
            total += labels.numel()
            correct += (predictions == labels.float()).sum().item()

            # Build absolute byte indices for each prediction in the batch.
            batch_offsets = start_offsets.cpu().numpy().astype(np.int64)
            batch_probs = probs.cpu().numpy().astype(np.float32)
            batch_labels = labels.cpu().numpy().astype(np.float32)
            batch_conf = confidence.cpu().numpy().astype(np.float32)
            sample_len = batch_probs.shape[1]
            local_positions = np.arange(sample_len, dtype=np.int64)
            abs_offsets = batch_offsets[:, None] + local_positions[None, :]

            # Safety mask for the last, possibly incomplete, coverage area.
            in_range = (abs_offsets >= 0) & (abs_offsets < bin_size)
            if not np.any(in_range):
                continue

            # Aggregate votes and labels into global buffers using advanced indexing with np.add.at to handle duplicates.
            flat_offsets = abs_offsets[in_range]
            flat_probs = batch_probs[in_range]
            flat_labels = batch_labels[in_range]
            flat_weights = np.clip(batch_conf[in_range], 1e-6, 1.0)

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
        invalid_count = int(np.count_nonzero(~valid_idx_mask))
        if invalid_count > 0:
            print(f"[ERROR] {invalid_count} offsets sont avant la première entrée meta ({meta_starts[0] if meta_starts else 'N/A'}).")

        # Error analysis by real type after global vote.
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
        percentage = (count / total_errors * 100) if total_errors > 0 else 0
        print(f"  Type: {data_type:<15} | Erreurs: {count:<5} | Proportion: {percentage:>6.2f}%")
    print(f"\n--- Évaluation terminée ---")
    print(f"Exactitude brute (avant vote): {accuracy:.2%} ({correct}/{total})")
    print(f"Exactitude agrégée (vote pondéré): {aggregated_accuracy:.2%} ({aggregated_correct}/{aggregated_total})")

    if visualizer:
        visualizer.saveJson(filepath=f"{cfg.VISUAL_EXPORT_DIR}/raid_evaluation_visualization.json")

if __name__ == "__main__":
    evaluate(genereateExport=True)