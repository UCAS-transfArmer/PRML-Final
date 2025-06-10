from tap import Tap
import torch
import os

class Args(Tap):
    # Experiment specific arguments
    project_name: str = 'PRML-Final'
    exp_name: str = 'vit-base16-imagenet-pretrain-bs512-original'
    
    # Dataset and model selection
    dataset: str = 'imagenet'
    model: str = 'vit'
    data_root: str = './data'  # 数据根目录
    
    # ViT specific arguments
    image_size: int = 224
    patch_size: int = 16
    dim: int = 768
    depth: int = 12
    heads: int = 12
    mlp_dim: int = 3072
    dropout: float = 0.0      # Dropout for ViT model
    use_mlp_head: bool = False

    # Training specific arguments
    ep: int = 300
    bs: int = 512
    lr: float = 3e-3          # Base learning rate (peak learning rate for the scheduler)
    weight_decay: float = 0.05 # Weight decay

    # Learning rate scheduler
    warmup_epochs: int = 5
    warmup_start_lr: float = 1e-6
    min_lr: float = 1e-5

    # Data augmentation parameters
    enhanced_augmentation: bool = True
    crop_padding: int = 28 # For CIFAR-10 - this might need adjustment if only imagenet is primary

    # ImageNet subset specific arguments for debugging
    imagenet_use_subset: bool = False
    imagenet_subset_num_classes: int = 10
    imagenet_subset_samples_per_class: int = 50
    imagenet_subset_val_samples_per_class: int = 10

    # System and logging
    num_workers: int = 8
    save_path: str = './ckpts/imagenet'
    keep_n_checkpoints: int = 2
    save_frequency: int = 20
    log_per_iter: int = 100
    device: str = 'cuda'
    use_data_parallel: bool = True # For multi-GPU training

    # --- Fine-tuning specific arguments ---
    pretrained_path: str = None # type: ignore
    head_lr_multiplier: float = 10.0
    freeze_backbone: bool = False
    grad_clip_norm: float = 1.0
    # Note: 'lr' default (3e-3) is now aligned with pre-training. 
    # Fine-tuning scripts should override this with a smaller value (e.g., 1e-4, 5e-5).

    # Mixed precision training
    use_amp: bool = True

    # Debugging
    debug_skip_real_data: bool = False #
    debug_use_fake_data: bool = False # 新增：用于从新工具文件加载伪造数据

    def process_args(self):
        """验证参数有效性"""
        # 模型验证
        if self.model not in ['logistic', 'boosting', 'resnet', 'vit']:
            raise ValueError(f"Model '{self.model}' not supported. "
                           f"Available options: logistic, boosting, resnet, vit")
        
        # 数据集验证 - 更新以支持imagenet
        if self.dataset not in ['cifar10', 'imagenet']:
            raise ValueError(f"Dataset '{self.dataset}' not supported. "
                           f"Available options: cifar10, imagenet")
        
        if self.dataset == 'imagenet' and self.imagenet_use_subset:
            if self.imagenet_subset_num_classes <= 0:
                raise ValueError("imagenet_subset_num_classes must be positive when using subset.")
            if self.imagenet_subset_samples_per_class <= 0:
                raise ValueError("imagenet_subset_samples_per_class must be positive when using subset.")
            if self.imagenet_subset_val_samples_per_class <= 0:
                raise ValueError("imagenet_subset_val_samples_per_class must be positive when using subset.")
        
        # 其他参数验证
        if self.bs <= 0:
            raise ValueError(f"Batch size must be positive, got {self.bs}")
        
        if self.ep <= 0:
            raise ValueError(f"Number of epochs must be positive, got {self.ep}")
            
        if self.lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {self.lr}")

        if self.warmup_start_lr < 0 or self.min_lr < 0:
            raise ValueError("warmup_start_lr and min_lr must be non-negative.")

        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative.")

        # 确保图像大小与patch_size兼容
        if self.model == 'vit' and self.image_size % self.patch_size != 0: # Check only for ViT
            raise ValueError(f"For ViT, image size ({self.image_size}) must be divisible by "
                           f"patch size ({self.patch_size})")

def get_args() -> Args: # 明确返回类型为 Args
    """使用 Tap 解析命令行参数并返回 Args 实例。"""
    args = Args().parse_args()
    args.process_args()  # 在参数解析后调用验证逻辑
    return args

if __name__ == '__main__':
    args_tap = get_args() # get_args 现在返回一个经过处理的 Args 实例
    print("\nParsed and processed arguments with Tap (get_args):")
    # 使用 as_dict() 获取 Tap 对象的字典表示形式以进行迭代
    for arg_name, arg_val in args_tap.as_dict().items():
        print(f"  {arg_name}: {arg_val}")
    