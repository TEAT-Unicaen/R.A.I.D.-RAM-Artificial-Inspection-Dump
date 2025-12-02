"""
Module defining the Patches class for image patch extraction.
Convolution, Kernel and Stride

TODO 

"""

import torch.nn as nn

class Patches(nn.Module):

    _imageSize: int
    _patchSize: int
    _nPatches: int
    _projector: nn.Conv2d

    def __init__(self, imageSize=224, kernelSize=16, inChannels=3, embedDim=768):
        super().__init__()
        self._imageSize = imageSize
        self._patchSize = kernelSize
        self._nPatches = (imageSize // kernelSize) ** 2

        #Conv2d -> Cuts and projects patches in one time
        #Channels : 3 -> RGB
        #outChannels : Dimension of the embedding space (numbers of vectors numbers per patch)
        #Kernel : size of the patch
        #Stride : step size for moving the kernel (equal to kernel size for non-overlapping patches)
        self._projector = nn.Conv2d(inChannels, embedDim, kernel_size=kernelSize, stride=kernelSize)

    def forward(self, image):
        #Im!age is an image tensor of shape : [1, 3, 224, 224] (batch_size, inChannels, height, width)
        image = self._projector(image)
        #Now image is of shape : [1, outChannels, patchesheight, patcheswidth] : it has become an array of patches
        #Then we Flatten height and width dimensions
        image = image.flatten(2)
        #Transpose to have (batch_size, n_patches, embedding_dim) because transformer expect this shape
        image = image.transpose(1, 2)
        #Final shape : [1, n_patches, embedding_dim]
        return image

    

