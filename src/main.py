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

    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
    total_batches = len(test_loader)
    print(f"Nombre d'échantillons à tester : {len(test_dataset)} (sur {total_batches} batchs)")


    print("Démarrage de l'analyse...")
    total, correct = 0, 0
    
    for batch_idx, (data, labels) in enumerate(test_loader):
        data, labels = data.to(device), labels.to(device)
        
        with torch.no_grad():
            outputs = model(data)
            
            probs = torch.sigmoid(outputs).squeeze()
            
            predictions = (probs > 0.5).float()
            
            correct_in_batch = (predictions == labels.float()).sum().item()
            
            total += labels.size(0)
            correct += correct_in_batch

            prob_sample = probs[0].item() if probs.dim() > 0 else probs.item()
            rep_ia = "Chiffré" if prob_sample > 0.5 else "Clair"
            vrai_rep = "Chiffré" if labels[0] == 1 else "Clair"
            
            status = "✅" if rep_ia == vrai_rep else "❌"

            print(f'Batch {batch_idx+1}/{total_batches} | : {rep_ia} : {vrai_rep} {status}')

if __name__ == "__main__":
    evaluate()