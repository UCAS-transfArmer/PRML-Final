import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Any
import os

def get_cifar10_dataloader(
    batch_size=256, 
    num_workers=4, 
    data_root='./data', 
    for_vit=True,
    image_size=224,
    crop_padding=28,
    enhanced_augmentation=False
):
    """
    创建 CIFAR-10 数据加载器
    Args:
        batch_size (int): 批次大小
        num_workers (int): 数据加载线程数
        data_root (str): 数据集根目录
        for_vit (bool): 是否使用 ViT 专用的数据增强
        image_size (int): 输入图像大小，用于 ViT
        crop_padding (int): 随机裁剪的填充大小
        enhanced_augmentation (bool): 是否使用增强的数据增强
    """
    # CIFAR-10 标准化参数
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    
    if for_vit:
        if enhanced_augmentation:
            # 增强的数据增强
            transform_train = transforms.Compose([
                transforms.Resize(image_size),  # 调整到指定大小
                transforms.RandomCrop(image_size, padding=crop_padding),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            # 标准的 ViT 数据增强
            transform_train = transforms.Compose([
                transforms.Resize(image_size),  # 调整到指定大小
                transforms.RandomCrop(image_size, padding=crop_padding),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        
        transform_test = transforms.Compose([
            transforms.Resize(image_size),  # 调整到指定大小
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
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
    
    return trainloader, testloader, 10, (3, image_size, image_size) if for_vit else (3, 32, 32)



def get_imagenet_dataloader(
    batch_size=256,  # 与 pretrain_vit_imagenet.sh 中的 --bs 对齐
    num_workers=8,   # 与 pretrain_vit_imagenet.sh 中的 --num_workers 对齐
    data_root='./data/imagenet',
    for_vit=True,
    image_size=224,  # 与 pretrain_vit_imagenet.sh 中的 --image_size 对齐
    crop_ratio=0.875,  # 标准 ImageNet 裁剪比例
    enhanced_augmentation=True  # 与 pretrain_vit_imagenet.sh 中的 --enhanced_augmentation 对齐
):
    """
    创建 ImageNet 数据加载器
    Args:
        batch_size (int): 批次大小
        num_workers (int): 数据加载线程数
        data_root (str): 数据集根目录
        for_vit (bool): 是否使用 ViT 专用的数据增强
        image_size (int): 输入图像大小，用于 ViT
        crop_ratio (float): 中心裁剪的比例
        enhanced_augmentation (bool): 是否使用增强的数据增强
    """
    # ImageNet 标准化参数
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    # 检查数据目录
    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        error_message = (
            f"ImageNet 数据目录结构不正确。期望的根目录 '{data_root}' 下应包含 'train' 和 'val' 子目录。\n"
            f"请从 http://image-net.org 手动下载 ImageNet ILSVRC2012 数据集，\n"
            f"并将其解压和组织成以下结构:\n"
            f"{data_root}/\n"
            f"├── train/\n"
            f"│   ├── n01440764/  # (类别ID，例如 tench)\n"
            f"│   │   ├── n01440764_10026.JPEG\n"
            f"│   │   └── ...\n"
            f"│   └── ... (其他类别)\n"
            f"└── val/\n"
            f"    ├── n01440764/\n"
            f"    │   ├── ILSVRC2012_val_00000293.JPEG\n"
            f"    │   └── ...\n"
            f"    └── ... (其他类别)\n"
            f"有关组织验证集的脚本，请参考常见的 PyTorch ImageNet 示例。"
        )
        raise ValueError(error_message)
    
    # 计算验证集的裁剪大小
    resize_size = int(image_size / crop_ratio)
    
    if for_vit:
        if enhanced_augmentation:
            # 增强的数据增强，适用于 ViT
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            # 标准的 ImageNet 增强, ViT 论文中使用
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        
        # 验证集转换
        transform_test = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        # 标准的 ImageNet 转换，非 ViT 模型
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    # 加载数据集
    trainset = torchvision.datasets.ImageFolder(
        root=train_dir,
        transform=transform_train
    )
    
    testset = torchvision.datasets.ImageFolder(
        root=val_dir,
        transform=transform_test
    )
    
    print(f"ImageNet 数据集加载完成。训练集: {len(trainset)} 样本, 验证集: {len(testset)} 样本")
    
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
    
    return trainloader, testloader, 1000, (3, image_size, image_size)




def get_dataloader(
    dataset_name, 
    batch_size, 
    num_workers=4, 
    data_root='./data', 
    for_vit=True, 
    enhanced_augmentation=False,
    image_size=224,
    crop_padding=28
):
    """
    获取指定数据集的数据加载器的工厂函数
    目前支持 CIFAR-10 和 ImageNet
    
    Args:
        dataset_name (str): 数据集名称 ('cifar10' 或 'imagenet')
        batch_size (int): 批次大小
        num_workers (int): 数据加载线程数
        data_root (str): 数据集根目录
        for_vit (bool): 是否使用 ViT 专用的数据增强
        enhanced_augmentation (bool): 是否使用增强的数据增强
        image_size (int): 输入图像大小，用于 ViT
        crop_padding (int): CIFAR-10 随机裁剪的填充大小
    
    Returns:
        tuple: (trainloader, testloader, num_classes, image_dims)
    """
    if dataset_name.lower() == 'cifar10':
        return get_cifar10_dataloader(
            batch_size, 
            num_workers, 
            data_root, 
            for_vit, 
            image_size, 
            crop_padding, 
            enhanced_augmentation
        )
    elif dataset_name.lower() == 'imagenet':
        # ImageNet 数据目录结构通常不同
        imagenet_root = os.path.join(data_root, 'imagenet')
        return get_imagenet_dataloader(
            batch_size,
            num_workers,
            imagenet_root,
            for_vit,
            image_size,
            crop_ratio=0.875,  # 标准 ImageNet 裁剪比例
            enhanced_augmentation=enhanced_augmentation
        )
    else:
        raise ValueError(f"数据集 '{dataset_name}' 暂不支持。目前仅支持 'cifar10' 和 'imagenet'。")




if __name__ == '__main__':
    # 测试用例
    print("Testing CIFAR-10 Dataloader (standard)...")
    try:
        train_loader, test_loader, n_classes, img_dims = get_dataloader('cifar10', 64, 2, for_vit=False)
        print(f"Classes: {n_classes}, Image Dims: {img_dims}")
        
        data_iter = iter(train_loader)
        images, labels = next(data_iter)
        print(f"Standard - Images batch shape: {images.shape}, Labels batch shape: {labels.shape}")
    except Exception as e:
        print(f"CIFAR-10 测试失败: {e}")
    
    # 测试 ViT 加载器
    print("\nTesting CIFAR-10 Dataloader (ViT)...")
    try:
        train_loader, test_loader, n_classes, img_dims = get_dataloader(
            'cifar10', 
            64, 
            2, 
            for_vit=True, 
            image_size=224,
            crop_padding=28
        )
        print(f"Classes: {n_classes}, Image Dims: {img_dims}")
        
        data_iter = iter(train_loader)
        images, labels = next(data_iter)
        print(f"ViT - Images batch shape: {images.shape}, Labels batch shape: {labels.shape}")
    except Exception as e:
        print(f"CIFAR-10 ViT 测试失败: {e}")
    
    # 测试 ImageNet 加载器（如果数据可用）
    print("\nTesting ImageNet Dataloader...")
    try:
        train_loader, test_loader, n_classes, img_dims = get_dataloader(
            'imagenet', 
            16,  # 使用小批量进行测试 
            2,
            for_vit=True
        )
        print(f"Classes: {n_classes}, Image Dims: {img_dims}")
        
        data_iter = iter(train_loader)
        images, labels = next(data_iter)
        print(f"ImageNet - Images batch shape: {images.shape}, Labels batch shape: {labels.shape}")
    except Exception as e:
        print(f"ImageNet 测试失败 (可能是因为数据集不可用): {e}")
