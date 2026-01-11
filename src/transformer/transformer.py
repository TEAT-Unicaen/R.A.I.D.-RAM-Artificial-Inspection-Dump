"""
RAM Dump Transformer module
This module contains classes and functions to define and manage transformer-based models for RAM dump analysis.

embedDim is 768 because its the most efficient dimension (original ViT paper)
embedDim must be divisible by numberHeads
depth is the number of transformer blocks stacked

CLS token is a special token, initialised at 0, added to the sequence of RAM dump patches to aggregate all information.

"""
import torch
import torch.nn as nn
from transformer.patches import Patches
from transformer.transformerBlock import TransformerBlock

class Transformer(nn.Module):

    _patches: Patches
    _clsToken: nn.Parameter
    _positionEmbed: nn.Parameter
    _blocks: nn.ModuleList
    _norm: nn.LayerNorm
    _head: nn.Linear


    def __init__(self, sequenceLength = 8192, kernelSize = 256, inChannels = 1, embedDim = 768, depth = 6, heads = 8, mlpRatio = 4, dropout = 0.5, numClasses = 2):
        """
        Initialize RAM Dump Transformer.
        
        Args:
            sequenceLength: Length of input RAM dump sequence (default: 8192 bytes)
            kernelSize: Size of each patch in bytes (default: 256 bytes)
            inChannels: Number of input channels (typically 1 for byte sequences)
            embedDim: Embedding dimension (must be divisible by heads)
            depth: Number of transformer blocks
            heads: Number of attention heads
            mlpRatio: MLP expansion ratio
            dropout: Dropout rate
            numClasses: Number of output classes for classification
        """
        super().__init__()

        #Patch extraction module for RAM dumps
        self._patches = Patches(sequenceLength, kernelSize, inChannels, embedDim)

        #CLS token initialization
        self._clsToken = nn.Parameter(torch.zeros(1, 1, embedDim))

        #Patches and CLS token positions
        self._positionEmbed = nn.Parameter(torch.zeros(1, 1 + self._patches._nPatches, embedDim))

        #Blocks stacking
        self._blocks = nn.ModuleList([
            TransformerBlock(embedDim, heads, mlpRatio, dropout) for _ in range(depth)
        ])

        #Final normalization layer
        self._norm = nn.LayerNorm(embedDim) #The CLS cleaner : normalisation before classification -> helps stabilize training and reduce noise
        self._head = nn.Linear(embedDim, numClasses) #The judge : final classification layer (if numclasses = 2 -> binary classification)


    def forward(self, x):
        """
        Forward pass through the RAM Dump Transformer.
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length] or [batch_size, 1, sequence_length]
               Represents raw bytes from RAM dumps
        
        Returns:
            Classification logits of shape [batch_size, numClasses]
        """
        #Preparing the data 
        x = self._patches(x)  # Extract patches from RAM dump

        #We make sure to expand the number of CLS tokens to match the batch size
        clsTokens = self._clsToken.expand(x.shape[0], -1, -1) 

        #Add CLS token at the beginning of each patch sequence
        x = torch.cat((clsTokens, x), dim=1)

        #Add position embedding to the patches
        x = x + self._positionEmbed

        #Pass through transformer blocks
        for index, block in enumerate(self._blocks):
            x = block(x)

        # Extract only the CLS token, throw all the other patches and normalize
        output = self._norm(x[:, 0])
        return self._head(output)







