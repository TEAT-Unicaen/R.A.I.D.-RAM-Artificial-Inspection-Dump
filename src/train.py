import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from torch.utils.data import DataLoader, random_split

from dumpManager.RamDumpDataset import RamDumpDataset

from transformers.bytesClassifier.BytesTransformerClassifier import BytesTransformerClassifier
import config as cfg


def resolve_checkpoint_path(checkpoint_name=None):
    """
    Resolve a training checkpoint path from an optional CLI argument.

    Accepted values for checkpoint_name:
    - None: start from scratch
    - simple file name (e.g. checkpoint_epoch_10.pt): searched in cfg.CHECKPOINT_DIR
    - relative path
    - absolute path
    """
    if not checkpoint_name:
        return None

    if os.path.isabs(checkpoint_name):
        return checkpoint_name

    if os.path.sep in checkpoint_name or "/" in checkpoint_name:
        return os.path.abspath(checkpoint_name)

    return os.path.join(cfg.CHECKPOINT_DIR, checkpoint_name)


def _unwrap_compiled_state_dict(state_dict):
    """
    Remove _orig_mod. prefix from compiled model state_dict keys.
    Useful when loading checkpoints saved with torch.compile.
    """
    unwrapped = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            unwrapped[key[10:]] = value
        else:
            unwrapped[key] = value
    return unwrapped

def compute_tv_loss(probs, device):
    """
    Compute Total Variation Loss using 1D convolution (GPU-optimized).
    
    Uses a fixed kernel [-1, 1] to compute adjacent differences efficiently.
    Much faster than manual slicing/subtraction on GPU.
    
    Args:
        probs: Tensor of shape (batch, length) - probability predictions
        device: torch device
    
    Returns:
        tv_loss: scalar tensor
    """
    # Reshape for conv1d: (batch, channels=1, length)
    probs_reshaped = probs.unsqueeze(1)
    
    # Kernel: [-1, 1] to compute differences (shape: 1, 1, 2)
    kernel = torch.tensor([[[-1.0, 1.0]]], device=device, dtype=probs.dtype)
    
    # Apply 1D convolution: computes probs[:, i+1] - probs[:, i]
    diff = F.conv1d(probs_reshaped, kernel, padding=0)
    
    # Compute mean absolute difference
    tv_loss = torch.mean(torch.abs(diff))
    
    return tv_loss

def train(
    learning_rate=cfg.TRAIN_CONFIG["learning_rate"],
    weight_decay=cfg.TRAIN_CONFIG["weight_decay"],
    num_epochs=cfg.TRAIN_CONFIG["num_epochs"],
    batch_size=cfg.TRAIN_CONFIG["batch_size"],
    checkpoint_name=None,
):

    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    if os.path.dirname(cfg.MODEL_PATH):
        os.makedirs(os.path.dirname(cfg.MODEL_PATH), exist_ok=True)

    full_dataset = RamDumpDataset(
        bin_path=cfg.BIN_PATH, 
        meta_path=cfg.META_PATH, 
        chunk_size=cfg.DATASET_CONFIG["chunk_size"],
        offset=cfg.DATASET_CONFIG["offset"]
    )

    # --- SPLIT TRAIN / VALIDATION (80% / 20%) ---
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # We use a fixed random seed for reproducibility of the split
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )

    # --- DATALOADERS ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **cfg.TRAIN_LOADER_CONFIG,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **cfg.VAL_LOADER_CONFIG, # We can use the same config for validation loader (num_workers, pin_memory, etc.)
    )

    # --- MODEL, CRITERION, OPTIMIZER, SCHEDULER ---
    model = BytesTransformerClassifier(**cfg.MODEL_CONFIG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()

    start_epoch = 0
    resumed_checkpoint_path = resolve_checkpoint_path(checkpoint_name)
    resumed_checkpoint = None
    if resumed_checkpoint_path:
        try:
            resumed_checkpoint = torch.load(resumed_checkpoint_path, map_location=device)
            if isinstance(resumed_checkpoint, dict):
                checkpoint_model_config = resumed_checkpoint.get("model_config", cfg.MODEL_CONFIG)
                checkpoint_state_dict = resumed_checkpoint.get("model_state_dict", resumed_checkpoint)

                if checkpoint_model_config != cfg.MODEL_CONFIG:
                    print("Avertissement: la config du modèle du checkpoint diffère de cfg.MODEL_CONFIG.")

                checkpoint_state_dict = checkpoint_state_dict if isinstance(checkpoint_state_dict, dict) else resumed_checkpoint
                checkpoint_state_dict = _unwrap_compiled_state_dict(checkpoint_state_dict)
                try:
                    load_result = model.load_state_dict(checkpoint_state_dict)
                except RuntimeError as load_error:
                    print(f"Avertissement: chargement strict impossible ({load_error}). Repli sur strict=False.")
                    load_result = model.load_state_dict(checkpoint_state_dict, strict=False)

                if getattr(load_result, "missing_keys", None):
                    print(f"Clés manquantes lors du chargement: {load_result.missing_keys}")
                if getattr(load_result, "unexpected_keys", None):
                    print(f"Clés inattendues lors du chargement: {load_result.unexpected_keys}")

                start_epoch = int(resumed_checkpoint.get("epoch", 0))
                print(f"Reprise de l'entraînement depuis : {resumed_checkpoint_path} (epoch {start_epoch})")
            else:
                print(f"Avertissement: le checkpoint '{resumed_checkpoint_path}' n'est pas au format attendu. Entraînement depuis zéro.")
        except FileNotFoundError:
            print(f"ERREUR CRITIQUE : Le checkpoint '{resumed_checkpoint_path}' est introuvable.")
            return
        except Exception as exc:
            print(f"ERREUR lors du chargement du checkpoint '{resumed_checkpoint_path}' : {exc}")
            return

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

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if isinstance(resumed_checkpoint, dict) and "optimizer_state_dict" in resumed_checkpoint:
        try:
            optimizer.load_state_dict(resumed_checkpoint["optimizer_state_dict"])
        except Exception as optimizer_error:
            print(f"Avertissement: impossible de restaurer l'optimizer ({optimizer_error}). Reprise sans son état.")

    #Scheduler to adjust Lr dynamically during training
    scheduler = None
    if cfg.SCHEDULER_CONFIG.get("enabled", False):
        scheduler_type = cfg.SCHEDULER_CONFIG.get("type", "cosine").lower()
        warmup_cfg = cfg.SCHEDULER_CONFIG.get("warmup", {})
        warmup_enabled = warmup_cfg.get("enabled", False)
        warmup_num_epochs = warmup_cfg.get("num_epochs", 3)

        # Skip warmup if resuming from checkpoint after warmup period
        if start_epoch > 0 and warmup_enabled and start_epoch >= warmup_num_epochs:
            warmup_enabled = False
            print(f"Warmup désactivé (reprise à epoch {start_epoch}, après warmup de {warmup_num_epochs} epochs)")
        
        # Create the main scheduler based on type
        if scheduler_type == "cosine":
            cosine_cfg = cfg.SCHEDULER_CONFIG.get("cosine", {})
            main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, cosine_cfg.get("T_max", num_epochs) - warmup_num_epochs - start_epoch) if warmup_enabled else max(1, cosine_cfg.get("T_max", num_epochs) - start_epoch),
                eta_min=cosine_cfg.get("eta_min", 0.0),
            )
        elif scheduler_type == "plateau":
            plateau_cfg = cfg.SCHEDULER_CONFIG.get("plateau", {})
            main_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=plateau_cfg.get("mode", "min"),
                factor=plateau_cfg.get("factor", 0.5),
                patience=plateau_cfg.get("patience", 2),
                min_lr=plateau_cfg.get("min_lr", 0.0),
            )
        else:
            raise ValueError(f"Type de scheduler non supporté: {scheduler_type}")
        
        # Wrap with warmup if enabled
        if warmup_enabled and scheduler_type != "plateau":
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=max(1, warmup_num_epochs - start_epoch),
            )
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[max(1, warmup_num_epochs - start_epoch)],
            )
        else:
            scheduler = main_scheduler

    # Use bf16 if supported for faster training and reduced memory usage
    useBf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    # Label smoothing 
    useLabelSmoothing = cfg.TRAIN_CONFIG.get("label_smoothing", False)

    print(f"--- Entraînement sur {device} ---")
    print(f"Bf16 activé: {useBf16}")
    print(f"Label smoothing activé: {useLabelSmoothing}")
    if scheduler is not None:
        print(f"Scheduler activé: {cfg.SCHEDULER_CONFIG.get('type', 'cosine')}")

    model.train()
    print("Démarrage de l'entraînement...")

    if start_epoch >= num_epochs:
        print(f"Le checkpoint chargé est déjà à l'epoch {start_epoch}, qui est >= num_epochs ({num_epochs}). Rien à entraîner.")
        return

    for epoch in range(start_epoch, num_epochs):
        total_loss = torch.zeros((), device=device)
        correct = torch.zeros((), device=device)
        total = 0
        start_time = time.time()
        for batch in train_loader:
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

                        if useLabelSmoothing:
                            y_valid = y[valid_mask].to(logits.dtype)
                            # --- LABEL SMOOTHING (0.1) ---
                            smoothing = 0.1
                            y_smoothed = torch.where(y_valid == 1.0, 1.0 - smoothing, smoothing)
                            loss = criterion(logits[valid_mask], y_smoothed)
                        else :
                            loss = criterion(logits[valid_mask], y[valid_mask].to(logits.dtype))
                    else:
                        loss = torch.tensor(0.0, device=device)

                    # Adding Total Variation Loss for smoother predictions (GPU-optimized with conv1d)
                    probs = torch.sigmoid(logits)
                    tv_loss = compute_tv_loss(probs, device)

                    # Lambda 0.1
                    combined_loss = loss + 0.1 * tv_loss
            else:
                logits = model(x)
                valid_mask = y >= 0
                
                if valid_mask.any():

                    if useLabelSmoothing:
                        y_valid = y[valid_mask].to(logits.dtype)
                        # --- LABEL SMOOTHING (0.1) ---
                        smoothing = 0.1
                        y_smoothed = torch.where(y_valid == 1.0, 1.0 - smoothing, smoothing)
                        loss = criterion(logits[valid_mask], y_smoothed)
                    else:
                        loss = criterion(logits[valid_mask], y[valid_mask].to(logits.dtype))
                else:
                    loss = torch.tensor(0.0, device=device)

                # Adding Total Variation Loss for smoother predictions (GPU-optimized with conv1d)
                probs = torch.sigmoid(logits)
                tv_loss = compute_tv_loss(probs, device)

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

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_loss_batches = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for v_x, v_y, _ in val_loader:
                v_x, v_y = v_x.to(device), v_y.to(device)
                
                if useBf16:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        v_logits = model(v_x)
                else:
                    v_logits = model(v_x)
                
                v_valid_mask = v_y >= 0
                if v_valid_mask.any():
                    if useLabelSmoothing:
                        v_y_valid = v_y[v_valid_mask].to(v_logits.dtype)
                        # --- LABEL SMOOTHING (0.1) ---
                        smoothing = 0.1
                        v_y_smoothed = torch.where(v_y_valid == 1.0, 1.0 - 0.1, 0.1)
                        v_loss = criterion(v_logits[v_valid_mask], v_y_smoothed)
                    else:
                        v_loss = criterion(v_logits[v_valid_mask], v_y[v_valid_mask].to(v_logits.dtype))
                    val_loss += v_loss.item()
                    val_loss_batches += 1
                    
                    v_preds = (torch.sigmoid(v_logits) > 0.5).float()
                    val_correct += (v_preds[v_valid_mask] == v_y[v_valid_mask]).sum().item()
                    val_total += v_valid_mask.sum().item()

        avg_val_loss = val_loss / val_loss_batches if val_loss_batches > 0 else 0
        val_accuracy = val_correct / val_total if val_total > 0 else 0

        avg_loss = total_loss.item() / len(train_loader) if len(train_loader) > 0 else 0
        accuracy = (correct / total).item() if total > 0 else 0.0
        
        # --- SCHEDULER UPDATE ---
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss) # On step sur la loss de VALIDATION
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        # --- PRINTING RESULTS ---
        print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Train Acc: {accuracy:.2%} | Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_accuracy:.2%} | LR: {current_lr:.2e} | Time: {end_time - start_time:.2f}s")
        
        model.train() # Repasser en mode train pour l'epoch suivante

        checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pt")
        # Always save unwrapped model state_dict, even if model is compiled
        state_dict = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "model_config": cfg.MODEL_CONFIG,
        }, checkpoint_path)
        print(f"Checkpoint sauvegardé : {checkpoint_path}")

    # Always save unwrapped model state_dict, even if model is compiled
    state_dict = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
    torch.save({
        "model_state_dict": state_dict,
        "model_config": cfg.MODEL_CONFIG,
        "train_config": cfg.TRAIN_CONFIG,
        "dataset_config": cfg.DATASET_CONFIG,
    }, cfg.MODEL_PATH)
    print(f"Modèle sauvegardé sous : {cfg.MODEL_PATH}")
    print("Entraînement terminé.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîne le modèle R.A.I.D.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Nom de checkpoint (dans checkpoints/), chemin relatif ou absolu. "
            "Si absent, l'entraînement démarre depuis zéro."
        ),
    )
    args = parser.parse_args()

    if os.path.exists(cfg.BIN_PATH) and os.path.exists(cfg.META_PATH):
        train(checkpoint_name=args.checkpoint)
    else:
        print("Erreur : Données introuvables. Lancez d'abord le générateur (DumpGenerator).")
