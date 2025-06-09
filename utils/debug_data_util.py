import torch
from torch.utils.data import Dataset, DataLoader
import os # 需要导入 os 来检查操作系统类型

class FakeDataset(Dataset):
    """
    一个生成随机图像和标签的伪造数据集类。
    """
    def __init__(self, num_samples, image_dims_tuple, num_classes, image_size_tuple):
        """
        初始化伪造数据集。
        Args:
            num_samples (int): 数据集中的样本总数。
            image_dims_tuple (tuple): 图像的维度，格式为 (C, H, W)，例如 (3, 224, 224)。
            num_classes (int): 数据集中的类别总数。
            image_size_tuple (tuple): 图像的高度和宽度，格式为 (H, W)，例如 (224, 224)。
                                      或者单个 int (假设 H=W)。
        """
        self.num_samples = num_samples
        self.image_dims_tuple = image_dims_tuple  # (C, H, W)
        self.num_classes = num_classes
        if isinstance(image_size_tuple, int):
            self.h, self.w = image_size_tuple, image_size_tuple
        else:
            self.h, self.w = image_size_tuple[0], image_size_tuple[1]
        
        # 确保维度一致性
        if self.image_dims_tuple[1] != self.h or self.image_dims_tuple[2] != self.w:
            raise ValueError(
                f"图像维度元组的高度/宽度 ({self.image_dims_tuple[1]},{self.image_dims_tuple[2]}) "
                f"必须与图像尺寸元组的高度/宽度 ({self.h},{self.w}) 相匹配。"
            )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        生成一个随机图像和标签。
        """
        # 生成 (C, H, W) 格式的随机图像张量
        image = torch.randn(self.image_dims_tuple[0], self.h, self.w)
        # 生成一个随机整数标签
        label = torch.randint(0, self.num_classes, (1,)).item()
        return image, label

def get_debug_dataloaders(args):
    """
    为调试目的创建并返回使用 FakeDataset 的 DataLoader 实例。
    它会模拟 trainloader, testloader, num_classes, 和 image_dims 的返回。
    """
    print("警告: [调试模式] 正在使用伪造数据加载器。")

    # 根据参数确定伪造数据的类别数
    if args.imagenet_use_subset:
        num_classes_debug = args.imagenet_subset_num_classes
        print(f"调试模式，ImageNet子集，伪造类别数: {num_classes_debug}")
    elif args.dataset.lower() == 'imagenet':
        num_classes_debug = 1000 # 完整 ImageNet 调试时伪造1000类
        print(f"调试模式，完整ImageNet，伪造类别数: {num_classes_debug}")
    else:
        num_classes_debug = getattr(args, 'num_classes', 10) # 对于其他数据集，尝试获取num_classes或默认为10
        print(f"调试模式，数据集 {args.dataset}，伪造类别数: {num_classes_debug}")


    image_channels = 3  # 标准彩色图像通常为3通道
    # image_size 来自 args，通常是一个整数
    image_height_width_tuple = (args.image_size, args.image_size)  # (H, W)
    # image_dims_debug_tuple 是 (C, H, W)
    image_dims_debug_tuple = (image_channels, args.image_size, args.image_size)

    # 为快速调试使用较少的伪造样本
    num_fake_train_samples = args.bs * 5  # 例如，5个批次的训练数据
    num_fake_test_samples = args.bs * 2   # 例如，2个批次的测试数据

    fake_train_dataset = FakeDataset(num_fake_train_samples, image_dims_debug_tuple, num_classes_debug, image_height_width_tuple)
    fake_test_dataset = FakeDataset(num_fake_test_samples, image_dims_debug_tuple, num_classes_debug, image_height_width_tuple)

    # 确定 num_workers，特别是在Windows上，对于简单数据集0可能更安全或更快
    # effective_num_workers = 0 if os.name == 'nt' and args.num_workers > 0 else args.num_workers
    effective_num_workers = 0 # 为了伪造数据的简单和快速，直接设为0

    trainloader = DataLoader(
        fake_train_dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=effective_num_workers,
        pin_memory=True # 如果设备是GPU，pin_memory可以加速数据传输
    )
    testloader = DataLoader(
        fake_test_dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=effective_num_workers,
        pin_memory=True
    )

    print(f"使用伪造数据: {len(fake_train_dataset)} 训练样本, {len(fake_test_dataset)} 测试样本, "
          f"{num_classes_debug} 类别, 图像维度 {image_dims_debug_tuple}")

    return trainloader, testloader, num_classes_debug, image_dims_debug_tuple