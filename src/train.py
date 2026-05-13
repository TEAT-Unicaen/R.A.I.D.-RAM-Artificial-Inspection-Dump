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
            if getattr(cfg, "COMPILE_MODE", None):
                compile_kwargs["mode"] = cfg.COMPILE_MODE
            model = torch.compile(model, **compile_kwargs)
        except Exception as exc:
            print(f"Compilation désactivée automatiquement: {exc}")
            print(f"Backends disponibles : {torch._dynamo.list_backends()}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    #Scheduler to adjust Lr dynamically during training
    scheduler = None
    if cfg.SCHEDULER_CONFIG.get("enabled", False):
        scheduler_type = cfg.SCHEDULER_CONFIG.get("type", "cosine").lower()
        if scheduler_type == "cosine":
            cosine_cfg = cfg.SCHEDULER_CONFIG.get("cosine", {})
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cosine_cfg.get("T_max", num_epochs),
                eta_min=cosine_cfg.get("eta_min", 0.0),
            )
        elif scheduler_type == "plateau":
            plateau_cfg = cfg.SCHEDULER_CONFIG.get("plateau", {})
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=plateau_cfg.get("mode", "min"),
                factor=plateau_cfg.get("factor", 0.5),
                patience=plateau_cfg.get("patience", 2),
                min_lr=plateau_cfg.get("min_lr", 0.0),
            )
        else:
            raise ValueError(f"Type de scheduler non supporté: {scheduler_type}")

    # Use bf16 if supported for faster training and reduced memory usage
    useBf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    print(f"--- Entraînement sur {device} ---")
    print(f"Bf16 activé: {useBf16}")
    if scheduler is not None:
        print(f"Scheduler activé: {cfg.SCHEDULER_CONFIG.get('type', 'cosine')}")

    model.train()
    print("Démarrage de l'entraînement...")

    for epoch in range(num_epochs):
        total_loss = torch.zeros((), device=device)
        correct = torch.zeros((), device=device)
        total = 0
        start_time = time.time()
        for batch in dataloader:
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            if useBf16:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                    logits = model(x)
                    valid_mask = y >= 0
                    
                    if valid_mask.any():
                        loss = criterion(logits[valid_mask], y[valid_mask])
                    else:
                        loss = torch.tensor(0.0, device=device)

                    # Adding Total Variation Loss for smoother predictions (on full sequence to preserve spatial structure)
                    probs = torch.sigmoid(logits)
                    tv_loss = torch.mean(torch.abs(probs[:, 1:] - probs[:, :-1]))

                    # Lambda 0.1
                    combined_loss = loss + 0.1 * tv_loss
            else:
                logits = model(x)
                valid_mask = y >= 0
                
                if valid_mask.any():
                    loss = criterion(logits[valid_mask], y[valid_mask])
                else:
                    loss = torch.tensor(0.0, device=device)

                # Adding Total Variation Loss for smoother predictions (on full sequence to preserve spatial structure)
                probs = torch.sigmoid(logits)
                tv_loss = torch.mean(torch.abs(probs[:, 1:] - probs[:, :-1]))

                # Lambda 0.1
                combined_loss = loss + 0.1 * tv_loss

            combined_loss.backward()
            optimizer.step()
            
            total_loss += loss.detach()
            with torch.no_grad():
                preds = (probs > 0.5).float()
                # Only count accuracy on labeled positions
                if valid_mask.any():
                    correct += (preds[valid_mask] == y[valid_mask]).sum()
                    total += valid_mask.sum().item()
                else:
                    total += 0
            
        end_time = time.time()
        avg_loss = (total_loss / len(dataloader)).item()
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_loss)
            else:
                scheduler.step()

        accuracy = (correct / total).item() if total > 0 else 0.0
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2%} | LR: {current_lr:.2e} | Time: {end_time - start_time:.2f}s")

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