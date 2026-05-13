import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import DataLoader

from dumpManager.RamDumpDataset import RamDumpDataset

from transformers.bytesClassifier.BytesTransformerClassifier import BytesTransformerClassifier
import config as cfg

def train(
    learning_rate=cfg.TRAIN_CONFIG["learning_rate"],
    weight_decay=cfg.TRAIN_CONFIG["weight_decay"],
    num_epochs=cfg.TRAIN_CONFIG["num_epochs"],
    batch_size=cfg.TRAIN_CONFIG["batch_size"],
):

    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    if os.path.dirname(cfg.MODEL_PATH):
        os.makedirs(os.path.dirname(cfg.MODEL_PATH), exist_ok=True)

    dataset = RamDumpDataset(
        bin_path=cfg.BIN_PATH, 
        meta_path=cfg.META_PATH, 
        chunk_size=cfg.DATASET_CONFIG["chunk_size"],
        offset=cfg.DATASET_CONFIG["offset"]
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.TRAIN_LOADER_CONFIG["num_workers"],
        pin_memory=cfg.TRAIN_LOADER_CONFIG["pin_memory"],
        prefetch_factor=cfg.TRAIN_LOADER_CONFIG["prefetch_factor"],
    )

    model = BytesTransformerClassifier(**cfg.MODEL_CONFIG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    if cfg.DO_COMPILE_MODEL and hasattr(torch, "compile"):
        try:
            print("Tentative de compilation du modèle pour une meilleure performance ...")
            compile_kwargs = {}
            if getattr(cfg, "COMPILE_BACKEND", None):
                compile_kwargs["backend"] = cfg.COMPILE_BACKEND
            model = torch.compile(model, **compile_kwargs)
        except Exception as exc:
            print(f"Compilation désactivée automatiquement: {exc}")
            print(f"Backends disponibles : {torch._dynamo.list_backends()}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    useBf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    print(f"--- Entraînement sur {device} ---")
    print(f"bf16 activé: {useBf16}")

    model.train()
    print("Démarrage de l'entraînement...")

    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        start_time = time.time()
        for batch in dataloader:
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            if useBf16:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                    logits = model(x)
                    loss = criterion(logits, y)

                    # Adding Total Variation Loss for smoother predictions
                    probs = torch.sigmoid(logits)
                    # Calculate the total variation loss by summing the absolute differences between adjacent probabilities
                    tv_loss = torch.mean(torch.abs(probs[:, 1:] - probs[:, :-1]))

                    # Lambda 0.1
                    combined_loss = loss + 0.1 * tv_loss
            else:
                logits = model(x)
                loss = criterion(logits, y)

                # Adding Total Variation Loss for smoother predictions
                probs = torch.sigmoid(logits)
                # Calculate the total variation loss by summing the absolute differences between adjacent probabilities
                tv_loss = torch.mean(torch.abs(probs[:, 1:] - probs[:, :-1]))

                # Lambda 0.1
                combined_loss = loss + 0.1 * tv_loss

            combined_loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.numel()
            
        end_time = time.time()
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(dataloader):.4f} | Acc: {accuracy:.2%} | Time: {end_time - start_time:.2f}s")

        checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "model_config": cfg.MODEL_CONFIG,
        }, checkpoint_path)
        print(f"Checkpoint sauvegardé : {checkpoint_path}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": cfg.MODEL_CONFIG,
        "train_config": cfg.TRAIN_CONFIG,
        "dataset_config": cfg.DATASET_CONFIG,
    }, cfg.MODEL_PATH)
    print(f"Modèle sauvegardé sous : {cfg.MODEL_PATH}")
    print("Entraînement terminé.")

if __name__ == "__main__":
    if os.path.exists(cfg.BIN_PATH) and os.path.exists(cfg.META_PATH):
        train()
    else:
        print("Erreur : Données introuvables. Lancez d'abord le générateur (DumpGenerator).")