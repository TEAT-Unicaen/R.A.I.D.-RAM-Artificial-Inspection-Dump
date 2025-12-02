"""
Transformer module for R.A.I.D. - RAM Artificial Inspection Dump
This module contains classes and functions to define and manage transformer-based models.
"""

import torch.nn as nn
from .mlp import MLP

class TransformerBlock(nn.Module):

    _norm1: nn.LayerNorm
    _attention: nn.Module

    _norm2: nn.LayerNorm


    def __init__(self, embedDim, numberHeads, mlpRatio = 4, dropout = 0.5):
        super().__init__()

        ###ATTENTION LAYER###
        #Normalization layer before attention
        self._norm1 = nn.LayerNorm(embedDim)

        #Multihead Attention Layer
        #Make attention with multiple heads to capture different metrics (eg. colors, shapes, etc.)  
        self._attention = nn.MultiheadAttention(embedDim, numberHeads, batch_first=True)
        #######################

        ###MLP LAYER###
        #Normalization layer before MLP
        self._norm2 = nn.LayerNorm(embedDim)

        #MLP Layer : Feedforward neural network to process the output of the attention mechanism
        self._mlp = MLP(embedDim, int(mlpRatio * embedDim), dropout)
        ################

    def forward(self, x):
        #X is of shape (batch_size, n_patches, embedDim)

        ## Block 1 : Attention
        xNormalised = self._norm1(x)
        #Sending xNormalised 3 times because its self attention (Q = K = V)
        attentionOutput, _ = self._attention(xNormalised, xNormalised, xNormalised)

        x = x + attentionOutput  # Residual connection

        ##Block 2 : MLP
        xNormalised = self._norm2(x)
        mlpOutput = self._mlp(xNormalised)
        x = x + mlpOutput  # Residual connection

        return x