import torch
import torchvision
import torchvision.transforms as transforms

def get_cifar10_dataloader(batch_size, num_workers, data_root='./data', for_vit=False,enhanced_augmentation=False):
    """
    Creates and returns CIFAR-10 train and test DataLoaders.

    Args:
        batch_size (int): Batch size for the DataLoaders.
        num_workers (int): Number of worker processes for data loading.
        data_root (str): Root directory to store/load the CIFAR-10 dataset.
        for_vit (bool): If True, applies transformations suitable for Vision Transformers 
                        (e.g., resizing to 224x224).

    Returns:
        tuple: (trainloader, testloader, num_classes, image_dims)
               num_classes (int): Number of classes in the dataset (10 for CIFAR-10).
               image_dims (tuple): Dimensions of an image tensor (C, H, W).
    """
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2023, 0.1994, 0.2010)
    num_classes = 10

    if for_vit:
        # Transformations for Vision Transformer (expects 224x224 images)
        image_size = 224
        if enhanced_augmentation:
            #some extra augmentations
            transform_train_list = [
                transforms.Resize(image_size, antialias=True),  # Resize
                transforms.RandomCrop(image_size, padding=int(image_size*0.125), pad_if_needed=True), 
                transforms.RandomHorizontalFlip(p=0.5),  
                transforms.RandomRotation(degrees=15),  
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color adjustments
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Random affine: translation and scaling
                transforms.ToTensor(),
                transforms.Normalize(cifar10_mean, cifar10_std),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),  # Random erasing, 50% chance
            ]
        else:
            # Standard transforms for ViT without enhanced augmentation
            transform_train_list = [
                transforms.Resize(image_size, antialias=True),  # Resize 
                transforms.RandomCrop(image_size, padding=int(image_size*0.125), pad_if_needed=True),  # Random crop
                transforms.RandomHorizontalFlip(p=0.5), 
                transforms.ToTensor(),
                transforms.Normalize(cifar10_mean, cifar10_std),
            ]
        transform_test_list = [
            transforms.Resize(image_size, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ]
        image_dims = (3, image_size, image_size)
    
    else:
        # Standard transforms for CNNs like ResNet (32x32)
        image_size = 32
        if enhanced_augmentation:
            transform_train_list = [
                transforms.RandomCrop(image_size, padding=4, padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(cifar10_mean, cifar10_std, inplace=True),
            ]
        else:
            transform_train_list = [
                transforms.ToTensor(),
                transforms.Normalize(cifar10_mean, cifar10_std, inplace=True), # Note: inplace=True might be unexpected if you reuse the tensor. Consider removing it or being aware of its effect.
            ]
        transform_test_list = [
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ]
        image_dims = (3, image_size, image_size)


    transform_train = transforms.Compose(transform_train_list)
    transform_test = transforms.Compose(transform_test_list)

    try:
        trainset = torchvision.datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=transform_train)
    except Exception as e:
        print(f"Failed to download/load CIFAR10 trainset. Trying with download=False (if already downloaded). Error: {e}")
        trainset = torchvision.datasets.CIFAR10(
            root=data_root, train=True, download=False, transform=transform_train)


    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)


    try:
        testset = torchvision.datasets.CIFAR10(
            root=data_root, train=False, download=True, transform=transform_test)
    except Exception as e:
        print(f"Failed to download/load CIFAR10 testset. Trying with download=False. Error: {e}")
        testset = torchvision.datasets.CIFAR10(
            root=data_root, train=False, download=False, transform=transform_test)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    
    return trainloader, testloader, num_classes, image_dims

def get_dataloader(dataset_name, batch_size, num_workers, data_root='./data', for_vit=False,enhanced_augmentation=False):
    """
    Factory function to get DataLoaders for a specified dataset.
    Currently only supports CIFAR-10.
    """
    if dataset_name.lower() == 'cifar10':
        return get_cifar10_dataloader(batch_size, num_workers, data_root, for_vit, enhanced_augmentation)
    else:
        raise ValueError(f"Dataset '{dataset_name}' is not currently supported.")

if __name__ == '__main__':

    print("Testing CIFAR-10 Dataloader (standard)...")
    train_loader, test_loader, n_classes, img_dims = get_dataloader('cifar10', 64, 2, for_vit=False)
    print(f"Classes: {n_classes}, Image Dims: {img_dims}")
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    print(f"Standard - Images batch shape: {images.shape}, Labels batch shape: {labels.shape}")
