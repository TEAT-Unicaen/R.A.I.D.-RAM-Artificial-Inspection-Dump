"""
Module implementing a simple Multi-Layer Perceptron (MLP) for use in transformer architectures.
mlpDim : Dimension of the hidden layer in the MLP -> must be larger of 4 times the embedDim to allow complex pattern learning
"""

import torch.nn as nn

class MLP(nn.Module):

    _fc1: nn.Linear
    _fc2: nn.Linear
    _activation: nn.GELU

    _dropout: nn.Dropout

    def __init__(self, embedDim, mlpDim, dropout = 0.5):
        super().__init__()
        #First fully connected layer -> We expand the dimension to have more "mental space" for the model to learn complex patterns
        self._fc1 = nn.Linear(embedDim, mlpDim)
        #Activation function (Gaussian Error Linear Unit) : helps introduce non-linearity and learning complex patterns
        self._activation = nn.GELU()
        #Dropout layer for regularization
        self._dropout = nn.Dropout(dropout)
        #Second fully connected layer : We reduce back to the original embedding dimension
        self._fc2 = nn.Linear(mlpDim, embedDim)

    def forward(self, x):
        #X is of shape (batch_size, n_patches, embedDim)

        x = self._fc1(x) 
        x = self._activation(x)
        x = self._dropout(x)
        x = self._fc2(x)
        x = self._dropout(x)
        return x