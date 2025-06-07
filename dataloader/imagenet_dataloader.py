import torch
import torchvision
import torchvision.transforms as transforms
from typing import Tuple
import os

def get_imagenet_dataloader(batch_size: int, num_workers: int, data_root: str = './data') -> Tuple:
    """
    Creates and returns ImageNet-1k train and test DataLoaders, specifically for ViT pretraining.

    Args:
        batch_size (int): Batch size for the DataLoaders.
        num_workers (int): Number of worker processes for data loading.
        data_root (str): Root directory to store/load the ImageNet-1k dataset.

    Returns:
        tuple: (trainloader, testloader, num_classes, image_dims)
               num_classes (int): Number of classes in the dataset (1000 for ImageNet-1k).
               image_dims (tuple): Dimensions of an image tensor (C, H, W).
    """
    # ImageNet-1k statistics for normalization
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    num_classes = 1000  # ImageNet-1k has 1000 classes
    image_size = 224  # Standard for ViT pretraining

    # Transformations for Vision Transformer pretraining
    transform_train_list = [
        transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(3/4, 4/3), antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ]
    transform_test_list = [
        transforms.Resize(256, antialias=True),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ]
    image_dims = (3, image_size, image_size)

    transform_train = transforms.Compose(transform_train_list)
    transform_test = transforms.Compose(transform_test_list)

    # Check if data_root exists
    if not os.path.exists(data_root):
        print(f"Warning: Data directory {data_root} does not exist. Creating it...")
        os.makedirs(data_root, exist_ok=True)

    # Load ImageNet-1k train dataset
    try:
        trainset = torchvision.datasets.ImageNet(
            root=data_root, split='train', download=True, transform=transform_train)
    except Exception as e:
        print(f"Failed to download/load ImageNet trainset. Trying with download=False (if already downloaded). Error: {e}")
        trainset = torchvision.datasets.ImageNet(
            root=data_root, split='train', download=False, transform=transform_train)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, persistent_workers=num_workers > 0, drop_last=True)

    # Load ImageNet-1k validation dataset
    try:
        testset = torchvision.datasets.ImageNet(
            root=data_root, split='val', download=True, transform=transform_test)
    except Exception as e:
        print(f"Failed to download/load ImageNet testset. Trying with download=False. Error: {e}")
        testset = torchvision.datasets.ImageNet(
            root=data_root, split='val', download=False, transform=transform_test)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, persistent_workers=num_workers > 0, drop_last=False)

    return trainloader, testloader, num_classes, image_dims

if __name__ == '__main__':
    print("Testing ImageNet-1k DataLoader for ViT pretraining...")
    try:
        train_loader, test_loader, n_classes, img_dims = get_imagenet_dataloader(
            batch_size=64, num_workers=2, data_root='./data')
        print(f"Classes: {n_classes}, Image Dims: {img_dims}")
        data_iter = iter(train_loader)
        images, labels = next(data_iter)
        print(f"ViT - Images batch shape: {images.shape}, Labels batch shape: {labels.shape}")
    except Exception as e:
        print(f"Error during testing: {e}")