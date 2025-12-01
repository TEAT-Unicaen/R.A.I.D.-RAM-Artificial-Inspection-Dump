from torchvision import transforms
from torchvision import datasets

def preprocessImagesDataset(dataSetPath):
    """
    Preprocess the images in the dataset located at dataSetPath.
    Applies resizing, center cropping, and conversion to tensor.
    Args:
        dataSetPath (str): Path to the dataset directory.
    Returns:
        dataset (torchvision.datasets.ImageFolder): Preprocessed image dataset.
    """

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(
        root=dataSetPath,       
        transform=preprocess 
    )

    return dataset