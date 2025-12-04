"""
This module is responsible for training the model.
It includes functions for loading datasets, defining the training loop, and evaluating model performance.

TODO : Add a scheduler for learning rate adjustment -> torch.optim.lr_scheduler ?

"""

import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.amp as amp
import random
import numpy as np

import utils.imagePreprocessor as ipp
from utils.config import TrainingConfig
from transformer.visionTransformer import VisionTransformer
from torch.cuda.amp import autocast, GradScaler
from datetime import datetime
import shutil
def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def trainModel(config: TrainingConfig = None):
    """
    Train the model with the given configuration.
    
    Args:
        config (TrainingConfig, optional): Training configuration. If None, loads from config.cfg
    """
    # Load configuration
    if config is None:
        config = TrainingConfig()
    
    # Set random seed for reproducibility
    set_seed(config.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load and preprocess dataset
    # point to dataset/train directory
    dataset_dir = os.path.join(os.getcwd(), "dataset", "train")
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    try:
        dataset = ipp.preprocessImagesDataset(dataset_dir)
    except TypeError:
        return RuntimeError("preprocessImagesDataset failed due to a bad dataset path.")
    dataLoader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, pin_memory=True, shuffle=True)

    # Create model and optimizer
    model = VisionTransformer(embedDim=config.embed_dim, dropout=config.dropout, depth=config.depth, heads=config.heads).to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=config.lr) #Best optimizer for vision transformers ATM
    criterion = nn.CrossEntropyLoss()
    scaler = amp.GradScaler(device)
    # Training loop
    print("Starting training...")
    for epoch in range(config.epochs):
        total = 0
        correct = 0
        for images, labels in dataLoader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            # Forward pass
            with amp.autocast(device):
                outputs = model.forward(images)
                loss = criterion(outputs, labels)

            # Backward pass and optimization
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            #Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{config.epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")

    print("Training complete.")
    path = f"output/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), f"{path}/shrekTransformerResult.pth")
    shutil.copy("config.cfg", path)