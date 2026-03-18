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
        offset=512 #no off for tests
    )

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

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(test_loader):
            data, labels = data.to(device), labels.to(device)

            logits = model(data)

            # Convert logits into probs
            probs = torch.sigmoid(logits) # 32*512 | batch_size * sequence_length -> Tensor
            predictions = (probs > 0.5).float() # batch_size * sequence_length

            # Global accuracy
            total += labels.numel()
            correct += (predictions == labels.float()).sum().item()

            # Details per sample
            batch_start = batch_idx * BATCH_SIZE
            batch_end = min(batch_start + data.size(0), len(test_dataset.samples))
            if batch_start >= batch_end:
                continue

            # Calculates the byte offset in the dataset for each sample in the batch
            effective_batch_size = batch_end - batch_start
            batch_offsets = np.fromiter(
                (test_dataset.samples[i][0] for i in range(batch_start, batch_end)),
                dtype=np.int64,
                count=effective_batch_size,
            )

            # Convert predictions and labels to CPU numpy arrays for analysis
            batch_preds = predictions[:effective_batch_size].cpu().numpy().astype(np.int8) # GPU --> CPU
            batch_labels = labels[:effective_batch_size].cpu().numpy().astype(np.int8)
            batch_correct = batch_preds == batch_labels

            # Each run of consecutive predictions (clear or crypted) is analyzed to determine its byte offset, size, correctness, and real type.
            sample_len = batch_preds.shape[1]
            transitions = (np.diff(batch_preds, axis=1) != 0) | (np.diff(batch_correct.astype(np.int8), axis=1) != 0)
            run_start_mask = np.zeros_like(batch_correct, dtype=bool)
            run_start_mask[:, 0] = True
            run_start_mask[:, 1:] = transitions

            # Flatten arrays to simplify run analysis across the entire batch
            flat_preds = batch_preds.reshape(-1)
            flat_correct = batch_correct.reshape(-1)
            run_starts_flat = np.flatnonzero(run_start_mask.reshape(-1))
            run_ends_flat = np.concatenate((run_starts_flat[1:], [flat_preds.size]))

            # For each run, calculate its byte offset, size, predicted class, correctness, and determine the real type based on metadata.
            segment_sizes = run_ends_flat - run_starts_flat
            sample_indices = run_starts_flat // sample_len
            offsets_in_sample = run_starts_flat % sample_len
            byte_offsets = batch_offsets[sample_indices] + offsets_in_sample
            run_preds = flat_preds[run_starts_flat]
            run_correct = flat_correct[run_starts_flat]

            # Determine the real type for each run based on the byte offset and metadata -> Label
            idxs = np.searchsorted(meta_starts_np, byte_offsets, side="right") - 1
            real_types = np.full(byte_offsets.shape, "unknown", dtype=object)

            # Validates that the found metadata index is correct (the byte offset falls within the ds-de range of the metadata entry). If not, the run is marked as "unknown" and an error is logged.
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
                    print(f"[ERROR] {out_of_range_count} offsets tombent dans un gap de metadata sur ce batch.")

            # Check for invalid metadata indices (offsets before the first metadata entry)
            invalid_count = int(np.count_nonzero(~valid_idx_mask))
            if invalid_count > 0:
                print(f"[ERROR] {invalid_count} offsets sont avant la première entrée meta ({meta_starts[0] if meta_starts else 'N/A'}).")

            # Error analysis: For runs that are incorrectly classified, we accumulate the total size of errors per real type to identify which types of data are most problematic for the model.
            error_mask = ~run_correct
            if np.any(error_mask):
                err_types = real_types[error_mask]
                err_sizes = segment_sizes[error_mask]
                unique_types, inverse_idx = np.unique(err_types, return_inverse=True)
                weighted_errors = np.bincount(inverse_idx, weights=err_sizes)
                for err_type, err_count in zip(unique_types, weighted_errors):
                    errorType[err_type] += int(err_count)

            if visualizer:
                for byte_offset, segment_size, run_pred, is_correct, real_type in zip(
                    byte_offsets,
                    segment_sizes,
                    run_preds,
                    run_correct,
                    real_types,
                ):
                    visualizer.addSegment(
                        offset=int(byte_offset),
                        size=int(segment_size),
                        prediction="crypted" if run_pred == 1 else "clear",
                        isCorrect=bool(is_correct),
                        trueLabel=real_type,
                    )

    accuracy = correct / total if total > 0 else 0
    print("\n--- Détails des erreurs par type de données ---")
    for data_type, count in sorted(errorType.items(), key=lambda x: -x[1]):
        print(f"  Type: {data_type} | Erreurs: {count}")
    print(f"\n--- Évaluation terminée ---")
    print(f"Exactitude totale : {accuracy:.2%} ({correct}/{total})")

    if visualizer:
        visualizer.saveJson(filepath=f"{cfg.VISUAL_EXPORT_DIR}/raid_evaluation_visualization.json")

if __name__ == "__main__":
    evaluate(genereateExport=True)