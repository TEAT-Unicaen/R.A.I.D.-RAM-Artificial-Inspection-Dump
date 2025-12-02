"""
This module is responsible for training the model.
It includes functions for loading datasets, defining the training loop, and evaluating model performance.

TODO : Add a scheduler for learning rate adjustment -> torch.optim.lr_scheduler ?

"""

import os
import torch
import torch.optim as optim
import torch.nn as nn

import utils.imagePreprocessor as ipp
from transformer.visionTransformer import VisionTransformer


def trainModel(epochs = 10):
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Metrics
    total = 0
    correct = 0

    # Load and preprocess dataset
    # point to dataset/train directory
    dataset_dir = os.path.join(os.getcwd(), "dataset", "train")
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    try:
        dataset = ipp.preprocessImagesDataset(dataset_dir)
    except TypeError:
        return RuntimeError("preprocessImagesDataset failed due to a bad dataset path.")
    dataLoader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # Create model and optimizer
    model = VisionTransformer().to(device)
    model.train()
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

            #Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")

    print("Training complete.")
    torch.save(model.state_dict(), "shrekTransformerResult.pth")



    