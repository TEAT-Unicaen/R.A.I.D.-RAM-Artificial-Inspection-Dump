"""
This module is responsible for training the model.
It includes functions for loading datasets, defining the training loop, and evaluating model performance.

TODO : Add a scheduler for learning rate adjustment -> torch.optim.lr_scheduler ?

"""

import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms

import imagePreprocessor
from transformer.visionTransformer import VisionTransformer


def trainModel(epochs = 10):
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load and preprocess dataset
    dataset = imagePreprocessor.preprocessImagesDataset("../dataset/train")
    dataLoader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # Create model and optimizer
    model = VisionTransformer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0003) #Best optimizer for vision transformers ATM
    criterion = nn.CrossEntropyLoss()

    # Training loop
    print("Starting training...")
    for epoch in range(epochs): 
        for images, labels in dataLoader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model.forward(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch [{epoch+1}/10], Loss: {loss.item():.4f}")

    print("Training complete.")



    