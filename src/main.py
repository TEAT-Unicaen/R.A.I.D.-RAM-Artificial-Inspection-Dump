import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys

# Tes imports personnalisés
from dumpManager.RamDumpDataset import RamDumpDataset
from transformers.bytesClassifier.BytesTransformerClassifier import BytesTransformerClassifier
import config as cfg

def evaluate():
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
        chunk_size=512
    )

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    total_batches = len(test_loader)
    print(f"Nombre d'échantillons à tester : {len(test_dataset)} (sur {total_batches} batchs)")

    print("Démarrage de l'analyse...")
    total, correct = 0, 0
    errorType = {}
    
    for batch_idx, (data, labels) in enumerate(test_loader):
        data, labels = data.to(device), labels.to(device)
        
        with torch.no_grad():
            outputs = model(data)
            probs = torch.sigmoid(outputs).squeeze()
            predictions = (probs > 0.5).float()
            
            correct_in_batch = (predictions == labels.float()).sum().item()
            total += labels.size(0)
            correct += correct_in_batch

            for i in range(len(labels)):
                pred = predictions[i].item()
                target = labels[i].item()
                
                if pred != target:
                    global_idx = batch_idx * test_loader.batch_size + i
                    offset_erreur, _ = test_dataset.samples[global_idx] #metadata file
                    
                    type_reel = "unknown"
                    for entry in test_dataset.metadata:
                        if entry['data_start'] <= offset_erreur < entry['data_end']:
                            type_reel = entry['type']
                            break
                    
                    #print(f"[ERROR] batch {batch_idx+1} ({i}) | Real type: {type_reel} | Predicted type: {'crypted' if pred == 1 else 'clear'}")
                    errorType[type_reel] = errorType.get(type_reel, 0) + 1

    accuracy = correct / total if total > 0 else 0
    print("\n--- Détails des erreurs par type de données ---")
    for data_type, count in errorType.items():
        print(f"Type: {data_type} | Erreurs: {count}")
    print(f"\n--- Évaluation terminée ---\nExactitude totale : {accuracy:.2%} ({correct}/{total})")

if __name__ == "__main__":
    evaluate()