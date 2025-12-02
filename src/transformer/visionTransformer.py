"""
Vision Transformer module
This module contains classes and functions to define and manage vision transformer-based models.

embedDim is 768 because its the most efficient dimension (original ViT paper)
embedDim must be divisible by numberHeads
depth is the number of transformer blocks stacked

CLS token is a special token, initialised at 0, added to the sequence of image patches to aggregate all information.

"""
import torch
import torch.nn as nn
from patches import Patches
from transformerBlock import TransformerBlock

class VisionTransformer:

    _patches: Patches
    _clsToken: nn.Parameter
    _positionEmbed: nn.Parameter
    _blocks: nn.ModuleList
    _norm: nn.LayerNorm
    _head: nn.Linear


    def __init__(self, imgSize = 224, kernelSize = 16, inChannels = 3, embedDim = 768, depth = 6, heads = 8, mlpRatio = 4, dropout = 0.5, numClasses = 2):
        super().__init__()

        #Patch extraction module
        self._patches = Patches(imgSize, kernelSize, inChannels, embedDim)

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
        #Preparing the data 
        x = self._patches(x)  # Extract patches

        #We make sure to expand the number of CLS tokens to match the batch size (because X is now multiples of images)
        clsTokens = self._clsToken.expand(x.shape[0], -1, -1) 

        #Add CLS token at the beginning of each patches list
        x = torch.cat((clsTokens, x), dim=1)

        #Add position embedding to the patches
        x = x + self._positionEmbed

        #Manual loop for ModuleList
        for index, block in enumerate(self._blocks):
            x = block(x)

        # Extract only the CLS token, throw all the other patches and normalize
        output = self._norm(x[:, 0])
        return self._head(output)







