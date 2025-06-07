import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Any

def get_cifar10_dataloader(batch_size=128, num_workers=2, data_root='./data', for_vit=True):
    """
    创建 CIFAR-10 数据加载器
    
    Args:
        batch_size (int): 批次大小
        num_workers (int): 数据加载线程数
        data_root (str): 数据集根目录
        for_vit (bool): 是否使用 ViT 专用的数据增强
    """
    # CIFAR-10 标准化参数
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    
    if for_vit:
        transform_train = transforms.Compose([
            transforms.Resize(224),  # 调整到224x224
            transforms.RandomCrop(224, padding=28),  # 添加padding
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        transform_test = transforms.Compose([
            transforms.Resize(224),  # 调整到224x224
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        # 非 ViT 模型的标准变换
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    # 加载数据集
    trainset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=transform_train
    )
    
    testset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=transform_test
    )
    
    # 创建数据加载器
    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return trainloader, testloader, 10, (3, 224, 224)  # 修改返回的图像维度为 224x224

def get_dataloader(dataset_name, batch_size, num_workers=4, data_root='./data', for_vit=True, enhanced_augmentation=False):
    """
    Factory function to get DataLoaders for a specified dataset.
    Currently only supports CIFAR-10.
    """
    if dataset_name.lower() == 'cifar10':
        return get_cifar10_dataloader(batch_size, num_workers, data_root, for_vit)
    else:
        raise ValueError(f"Dataset '{dataset_name}' is not currently supported.")

if __name__ == '__main__':

    print("Testing CIFAR-10 Dataloader (standard)...")
    train_loader, test_loader, n_classes, img_dims = get_dataloader('cifar10', 64, 2, for_vit=False)
    print(f"Classes: {n_classes}, Image Dims: {img_dims}")
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    print(f"Standard - Images batch shape: {images.shape}, Labels batch shape: {labels.shape}")
