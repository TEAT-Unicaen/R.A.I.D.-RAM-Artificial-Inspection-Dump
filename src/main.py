import torch
from torch.utils.data import DataLoader
import sys

from bisect import bisect_right
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
            probs = torch.sigmoid(logits)
            predictions = (probs > 0.5).float()

            total += labels.numel()
            correct += (predictions == labels.float()).sum().item()

            batch_preds = predictions.cpu().numpy() #GPU --> CPU
            batch_labels = labels.cpu().numpy()

            for i in range(len(data)):
                global_idx = batch_idx * BATCH_SIZE + i
                if global_idx >= len(test_dataset.samples):
                    break

                offset_val, _ = test_dataset.samples[global_idx]
                sample_preds = batch_preds[i]
                sample_labels = batch_labels[i]

                current_pos = 0
                sample_len = len(sample_preds)
                
                while current_pos < sample_len:
                    run_pred = sample_preds[current_pos]
                    run_label = sample_labels[current_pos]
                    run_correct = (run_pred == run_label)

                    run_end = current_pos + 1
                    while run_end < sample_len:
                        if (sample_preds[run_end] == run_pred) and ((sample_preds[run_end] == sample_labels[run_end]) == run_correct):
                            run_end += 1
                        else:
                            break
                    
                    byte_offset = offset_val + current_pos
                    segment_size = run_end - current_pos

                    idx = bisect_right(meta_starts, byte_offset) - 1 #recherche bin juste
                    real_type = "unknown"
                    if idx >= 0:
                        entry = test_dataset.metadata[idx]
                        if entry['ds'] <= byte_offset < entry['de']:
                            real_type = entry['t']
                    
                    if not run_correct:
                        errorType[real_type] += segment_size

                    if visualizer:
                        visualizer.addSegment(
                            offset=byte_offset,
                            size=segment_size,
                            prediction="crypted" if run_pred == 1 else "clear",
                            isCorrect=bool(run_correct),
                            trueLabel=real_type,
                        )
                    
                    current_pos = run_end

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