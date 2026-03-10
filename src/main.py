import torch
from torch.utils.data import DataLoader
import sys

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

    BATCH_SIZE = 32
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    total_batches = len(test_loader)
    print(f"Nombre d'échantillons à tester : {len(test_dataset)} (sur {total_batches} batchs)")

    visualizer = RaidVisualizerExporter() if genereateExport else None

    print("Démarrage de l'analyse...")
    total, correct = 0, 0
    errorType = {}

    for batch_idx, (data, labels) in enumerate(test_loader):
        data, labels = data.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(data)
            probs = torch.sigmoid(outputs)
            predictions = (probs > 0.5).float()

            total += labels.numel()
            correct += (predictions == labels.float()).sum().item()

            for i in range(len(labels)):
                pred_row = predictions[i]
                target_row = labels[i]
                isCorrect = (pred_row == target_row.float()).all().item()

                global_idx = batch_idx * BATCH_SIZE + i
                if global_idx >= len(test_dataset.samples):
                    continue

                offset_val, _ = test_dataset.samples[global_idx]

                real_type = "unknown"
                for entry in test_dataset.metadata:
                    if entry['ds'] <= offset_val < entry['de']:
                        real_type = entry['t']
                        break
                if real_type == "unknown":
                    print(f"Attention : Type de données inconnu pour l'offset {offset_val}")

                if not isCorrect:
                    errorType[real_type] = errorType.get(real_type, 0) + 1

                if visualizer:
                    majority_pred = "crypted" if pred_row.mean().item() > 0.5 else "clear" # TODO conversion en mean mais a retirer ça
                    visualizer.addSegment(
                        offset=offset_val,
                        size=test_dataset.chunk_size,
                        prediction=majority_pred,
                        isCorrect=isCorrect,
                        trueLabel=real_type,
                    )

    accuracy = correct / total if total > 0 else 0
    print("\n--- Détails des erreurs par type de données ---")
    for data_type, count in sorted(errorType.items(), key=lambda x: -x[1]):
        print(f"  Type: {data_type} | Erreurs: {count}")
    print(f"\n--- Évaluation terminée ---")
    print(f"Exactitude totale : {accuracy:.2%} ({correct}/{total})")

    if visualizer:
        visualizer.saveJson(filepath="raid_evaluation_visualization.json")

if __name__ == "__main__":
    evaluate(genereateExport=True)