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
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.CenterCrop(224),

        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(
        root=dataSetPath,       
        transform=preprocess 
    )

    return dataset