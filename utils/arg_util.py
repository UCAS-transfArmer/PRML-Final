from tap import Tap
import torch
import argparse

class Args(Tap):
    # Experiment specific arguments
    project_name: str = 'prml-final'
    exp_name: str = 'default_experiment'  # Name of the experiment, used for logging and saving checkpoints.
    
    dataset: str = 'cifar10'  # Dataset to use. Currently only cifar10 is supported.
    
    # Model specific arguments
    model: str = 'vit'  # Model architecture (choices: logistic, boosting, resnet, vit)
    
    # ViT specific arguments
    patch_size: int = 16  # Patch size for ViT (e.g., ViT-B/16)
    num_classes: int = 10  # Number of classes for classification (default for CIFAR-10)
    dim: int = 768  # Hidden dimension for ViT
    depth: int = 12  # Number of Transformer layers in ViT
    heads: int = 12  # Number of attention heads in ViT
    mlp_dim: int = 3072  # MLP dimension in ViT
    dropout: float = 0.1  # Dropout rate for ViT
    use_mlp_head: bool = True  # Whether to use an MLP head in ViT
    
    # Training specific arguments
    ep: int = 200  # Number of epochs
    tblr: float = 3e-4  # Initial learning rate for bs = 1024
    bs: int = 256  # Batch size
    bs_base: int = 256  # Base batch size for scaling learning rate
    
    # Logging and Saving
    log_per_iter: int = 10  # Log training progress every N iterations
    save_per_iter: int = 1000  # Save model checkpoint every N iterations (0 to disable)
    save_path: str = 'ckpts'  # Path to save checkpoints
    
    # System specific arguments
    num_workers: int = 4  # Number of dataloader workers
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'  # Device to train on

    def process_args(self):
        if self.model not in ['logistic', 'boosting', 'resnet', 'vit']:
            raise ValueError(f"Model {self.model} not supported")
        
        if self.dataset != 'cifar10':
            raise ValueError(f"Dataset {self.dataset} not supported. Currently only cifar10 is supported.")

def get_args():
    parser = argparse.ArgumentParser()
    
    # 基础配置
    parser.add_argument('--model', type=str, default='vit',
                      help='模型类型 (vit/resnet/logistic/boosting)')
    parser.add_argument('--dataset', type=str, default='cifar10',
                      help='数据集名称')
    
    # 设备配置
    parser.add_argument('--device', type=str, 
                      default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='运行设备 (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='数据加载的线程数')
    
    # ViT模型参数
    parser.add_argument('--image_size', type=int, default=32,
                      help='输入图像大小')
    parser.add_argument('--patch_size', type=int, default=4,
                      help='patch大小')
    parser.add_argument('--dim', type=int, default=384,
                      help='transformer特征维度')
    parser.add_argument('--depth', type=int, default=6,
                      help='transformer层数')
    parser.add_argument('--heads', type=int, default=6,
                      help='注意力头数')
    parser.add_argument('--mlp_dim', type=int, default=1536,
                      help='MLP隐层维度')
    parser.add_argument('--dropout', type=float, default=0.1,
                      help='dropout比率')
    parser.add_argument('--use_mlp_head', action='store_true', default=True,
                      help='是否使用MLP分类头')
    
    # 训练参数
    parser.add_argument('--bs', type=int, default=256,
                      help='批次大小')
    parser.add_argument('--ep', type=int, default=200,
                      help='训练轮数')
    parser.add_argument('--tblr', type=float, default=3e-4,
                      help='最大学习率（峰值）')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                      help='预热轮数')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                      help='最小学习率')
    parser.add_argument('--warmup_start_lr', type=float, default=1e-6,
                      help='预热起始学习率')
    parser.add_argument('--bs_base', type=int, default=256,
                      help='学习率缩放的基准批次大小')
    
    # 保存和日志参数
    parser.add_argument('--save_path', type=str, default='./ckpts',
                      help='模型保存路径')
    parser.add_argument('--exp_name', type=str, default='vit-cifar10-baseline',
                      help='实验名称')
    parser.add_argument('--project_name', type=str, default='PRML-Final',
                      help='wandb项目名称')
    parser.add_argument('--keep_n_checkpoints', type=int, default=3,
                      help='保留最新的N个检查点')
    parser.add_argument('--save_frequency', type=int, default=20,
                      help='每N个epoch保存一次模型')
    parser.add_argument('--log_per_iter', type=int, default=100,
                      help='每N步记录一次日志')
    parser.add_argument('--save_per_iter', type=int, default=1000,
                      help='每N步保存一次检查点')
    
    
    return parser.parse_args()

def validate_args(args):
    """验证参数有效性"""
    required_attrs = [
        'model', 'dataset', 'device', 'num_workers',
        'image_size', 'patch_size', 'dim', 'depth', 'heads',
        'mlp_dim', 'dropout', 'use_mlp_head',
        'bs', 'ep', 'tblr', 'bs_base',
        'save_path', 'exp_name', 'project_name',
        'keep_n_checkpoints', 'save_frequency',
        'log_per_iter', 'save_per_iter'
    ]
    
    missing_attrs = []
    for attr in required_attrs:
        if not hasattr(args, attr):
            missing_attrs.append(attr)
    
    if missing_attrs:
        raise ValueError(f"缺少以下参数: {', '.join(missing_attrs)}")
    
    return True

if __name__ == '__main__':
    args = get_args()
    print("Parsed arguments:")
    for arg_name, arg_val in vars(args).items():
        print(f"  {arg_name}: {arg_val}")
    
    print(f"\nRunning {args.model} model on {args.dataset} for {args.ep} epochs")
    print(f"Using device: {args.device}")