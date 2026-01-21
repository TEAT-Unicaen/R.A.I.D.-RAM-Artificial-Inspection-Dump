import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dumpManager.RamDumpDataset import RamDumpDataset

from transformers.bytesClassifier.BytesTransformerClassifier import BytesTransformerClassifier

import config as cfg

def train(learning_rate=0.001, num_epochs=5):
    dataset = RamDumpDataset(
        bin_path=cfg.BIN_PATH, 
        meta_path=cfg.META_PATH, 
        chunk_size=512
    )

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = BytesTransformerClassifier(dim_model=128, num_heads=4, num_layers=2, dropout=0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    print("Démarrage de l'entraînement...")

    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for x, y in dataloader:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            
            logits = model(x)
            
            loss = criterion(logits, y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)
            
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(dataloader):.4f} | Acc: {correct/total:.2%}")

    torch.save(model.state_dict(), cfg.MODEL_PATH)
    print(f"Modèle sauvegardé sous : {cfg.MODEL_PATH}")
    print("Entraînement terminé.")

if __name__ == "__main__":
    import os
    if os.path.exists(cfg.BIN_PATH) and os.path.exists(cfg.META_PATH):
        train(num_epochs=25)
    else:
        print("Erreur : Données introuvables. Lancez d'abord le générateur (DumpGenerator).")