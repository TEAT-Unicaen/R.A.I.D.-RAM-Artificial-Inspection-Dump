"""
Transformer module for R.A.I.D. - RAM Artificial Inspection Dump
This module contains classes and functions to define and manage transformer-based models.
"""

import torch.nn as nn

class Transformer(nn.Module):

    _attention: nn.Module

    def __init__(self, embedDim, numberHeads):
        super().__init__()

        #Multihead Attention Layer
        #Make attention with multiple heads to capture different metrics (eg. colors, shapes, etc.)  
        self._attention = nn.MultiheadAttention(embedDim, numberHeads, batch_first=True)

        #TODO : Add more layers 

        #TODO : Nomalization ?

    def forward(self, x):

        # TODO : Atention + residual connection ?

        return x