from tap import Tap
import torch
import os # 确保导入 os 如果要在 process_args 中使用

class Args(Tap):
    # Experiment specific arguments
    project_name: str = 'PRML-Final'
    exp_name: str = 'vit-base16-cifar'
    
    # Dataset and model selection
    dataset: str = 'cifar10'
    model: str = 'vit'
    data_root: str = './data'  # 数据根目录
    
    # ViT specific arguments
    image_size: int = 224
    patch_size: int = 16
    dim: int = 768
    depth: int = 12
    heads: int = 12
    mlp_dim: int = 3072
    dropout: float = 0.1      # Dropout for ViT model
    use_mlp_head: bool = False

    # Training specific arguments
    ep: int = 100
    bs: int = 128
    lr: float = 3e-4          # Base learning rate (peak learning rate for the scheduler)

    # Learning rate scheduler
    warmup_epochs: int = 10
    warmup_start_lr: float = 1e-6
    min_lr: float = 1e-6

    # Data augmentation parameters
    enhanced_augmentation: bool = False
    crop_padding: int = 28 # For CIFAR-10

    # ImageNet subset specific arguments for debugging
    imagenet_use_subset: bool = False
    imagenet_subset_num_classes: int = 10
    imagenet_subset_samples_per_class: int = 50
    imagenet_subset_val_samples_per_class: int = 10

    # System and logging
    num_workers: int = 4
    save_path: str = './ckpts'
    keep_n_checkpoints: int = 2
    save_frequency: int = 40
    log_per_iter: int = 100
    device: str = 'cuda'

    # --- Fine-tuning specific arguments ---
    pretrained_path: str = None
    head_lr_multiplier: float = 10.0
    freeze_backbone: bool = False
    grad_clip_norm: float = 1.0
    # Note: 'lr' default (3e-4) is for pre-training. 
    # Fine-tuning scripts should override this with a smaller value (e.g., 1e-4, 5e-5).

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

def get_args():
    """获取命令行参数"""
    args = Args().parse_args()
    args.process_args()
    return args

if __name__ == '__main__':
    args = get_args()
    print("Parsed arguments:")
    for arg_name, arg_val in vars(args).items():
        print(f"  {arg_name}: {arg_val}")
    
    print(f"\nRunning {args.model} model on {args.dataset} for {args.ep} epochs with base LR: {args.lr}")
    print(f"Using device: {args.device}")
    if args.pretrained_path:
        print(f"Attempting to load pretrained model from: {args.pretrained_path}")