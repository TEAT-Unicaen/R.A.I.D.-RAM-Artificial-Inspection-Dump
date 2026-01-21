"""
Module defining the Patches class for 1D RAM dump patch extraction.
Processes sequential binary data (RAM dumps) and converts it into patches for transformer processing.

"""

import torch.nn as nn

class Patches(nn.Module):

    _sequenceLength: int
    _patchSize: int
    _nPatches: int
    _projector: nn.Linear

    def __init__(self, sequenceLength=8192, kernelSize=256, inChannels=1, embedDim=768):
        """
        Initialize patch extraction for 1D RAM dump sequences.
        
        Args:
            sequenceLength: Length of the input RAM dump sequence (in bytes)
            kernelSize: Size of each patch (number of bytes per patch)
            inChannels: Number of input channels (typically 1 for byte sequences)
            embedDim: Dimension of the embedding space
        """
        super().__init__()
        self._sequenceLength = sequenceLength
        self._patchSize = kernelSize
        self._nPatches = sequenceLength // kernelSize
        
        # Linear projection: projects each patch from (kernelSize * inChannels) to embedDim
        # This replaces Conv2d for 1D sequential data
        self._projector = nn.Linear(kernelSize * inChannels, embedDim)

    def forward(self, dumpData):
        """
        Extract and project patches from 1D RAM dump sequence.
        
        Args:
            dumpData: Tensor of shape [batch_size, sequence_length] or [batch_size, inChannels, sequence_length]
                     Represents raw bytes from RAM dumps
        
        Returns:
            Tensor of shape [batch_size, n_patches, embedding_dim]
        """
        # Handle different input shapes
        if dumpData.dim() == 3:
            # Shape: [batch_size, inChannels, sequence_length]
            dumpData = dumpData.transpose(1, 2)  # -> [batch_size, sequence_length, inChannels]
        elif dumpData.dim() == 2:
            # Shape: [batch_size, sequence_length]
            dumpData = dumpData.unsqueeze(2)  # -> [batch_size, sequence_length, 1]
        
        batch_size = dumpData.shape[0]
        
        # Reshape to patches: [batch_size, n_patches, patch_size * inChannels]
        dumpData = dumpData.reshape(batch_size, self._nPatches, self._patchSize * dumpData.shape[2])
        
        # Project patches to embedding dimension
        dumpData = self._projector(dumpData)
        
        # Final shape: [batch_size, n_patches, embedding_dim]
        return dumpData