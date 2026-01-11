"""
This module is responsible for training the model.
It includes functions for loading RAM dump datasets, defining the training loop, and evaluating model performance.

TODO : Add a scheduler for learning rate adjustment -> torch.optim.lr_scheduler ?

"""

import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.amp as amp
import random
import numpy as np

import utils.dumpPreprocessor as dpp
from utils.config import TrainingConfig
from transformer.transformer import Transformer
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
    
    # Load and preprocess RAM dump dataset
    dataset_dir = os.path.join(os.getcwd(), config.data_dir)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    # Check if there are .bin files in the directory
    bin_files = [f for f in os.listdir(dataset_dir) if f.endswith('.bin')]
    if not bin_files:
        raise FileNotFoundError(f"No .bin dump files found in: {dataset_dir}")
    
    print(f"Loading RAM dump dataset from: {dataset_dir}")
    print(f"Found {len(bin_files)} dump file(s)")
    
    try:
        dataset = dpp.preprocessDumpDataset(
            dataset_dir, 
            sequence_length=config.sequence_length,
            exclude_noise=config.exclude_noise
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load dump dataset: {e}")
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    dataLoader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        pin_memory=True, 
        shuffle=True
    )

    # Create model for RAM dump analysis
    model = Transformer(
        sequenceLength=config.sequence_length,
        kernelSize=config.patch_size,
        inChannels=1,
        embedDim=config.embed_dim, 
        dropout=config.dropout, 
        depth=config.depth, 
        heads=config.heads,
        numClasses=config.num_classes
    ).to(device)
    
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    
    # Use weighted cross entropy for imbalanced classes
    try:
        class_weights = dpp.getClassWeights(dataset_dir, exclude_noise=config.exclude_noise)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print("Using weighted cross entropy loss for imbalanced classes")
    except:
        criterion = nn.CrossEntropyLoss()
        print("Using standard cross entropy loss")
    
    scaler = amp.GradScaler(device)
    
    # Training loop
    print("Starting training...")
    print(f"Sequence length: {config.sequence_length}, Patch size: {config.patch_size}")
    print(f"Number of classes: {config.num_classes}")
    
    for epoch in range(config.epochs):
        total = 0
        correct = 0
        epoch_loss = 0.0
        num_batches = 0
        
        for dump_data, labels in dataLoader:
            dump_data = dump_data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            
            # Forward pass
            with amp.autocast(device):
                outputs = model.forward(dump_data)
                loss = criterion(outputs, labels)

            # Backward pass and optimization
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            epoch_loss += loss.item()
            num_batches += 1
            
        accuracy = 100 * correct / total if total > 0 else 0
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch [{epoch+1}/{config.epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    print("Training complete.")
    path = f"output/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), f"{path}/ramDumpTransformer.pth")
    shutil.copy("config.cfg", path)
    print(f"Model saved to: {path}/ramDumpTransformer.pth")