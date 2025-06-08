from tap import Tap
import torch

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
    dropout: float = 0.1
    use_mlp_head: bool = False  # 是否使用MLP分类头
    
    # Training specific arguments
    ep: int = 100
    bs: int = 128  
    tblr: float = 3e-4  
    
    # Learning rate scheduler
    warmup_epochs: int = 10
    warmup_start_lr: float = 1e-6  
    min_lr: float = 1e-6  
    
    # 数据增强参数
    enhanced_augmentation: bool = False
    crop_padding: int = 28
    
    # System and logging
    num_workers: int = 4
    save_path: str = './ckpts'
    keep_n_checkpoints: int = 2
    save_frequency: int = 40
    log_per_iter: int = 100
    device: str = 'cuda'  # 默认使用CUDA
    
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
        
        # 其他参数验证
        if self.bs <= 0:
            raise ValueError(f"Batch size must be positive, got {self.bs}")
        
        if self.ep <= 0:
            raise ValueError(f"Number of epochs must be positive, got {self.ep}")
            
        # 确保图像大小与patch_size兼容
        if self.image_size % self.patch_size != 0:
            raise ValueError(f"Image size ({self.image_size}) must be divisible by "
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
    
    print(f"\nRunning {args.model} model on {args.dataset} for {args.ep} epochs")
    print(f"Using device: {args.device}")