from tap import Tap
import torch

class Args(Tap):
    # Experiment specific arguments
    project_name: str = 'PRML-Final'
    exp_name: str = 'vit-base16-cifar'
    
    # Dataset and model selection
    dataset: str = 'cifar10'
    model: str = 'vit'
    
    # ViT specific arguments
    image_size: int = 224
    patch_size: int = 16
    dim: int = 768
    depth: int = 12
    heads: int = 12
    mlp_dim: int = 3072
    dropout: float = 0.1
    
    # Training specific arguments
    ep: int = 100
    bs: int = 1024
    tblr: float = 1e-2
    
    # Learning rate scheduler
    warmup_epochs: int = 10
    warmup_start_lr: float = 1e-5
    min_lr: float = 1e-5
    
    # System and logging
    num_workers: int = 4
    save_path: str = './ckpts'
    keep_n_checkpoints: int = 2
    save_frequency: int = 40
    log_per_iter: int = 100
    save_per_iter: int = 1000
    
    def process_args(self):
        if self.model not in ['logistic', 'boosting', 'resnet', 'vit']:
            raise ValueError(f"Model {self.model} not supported")
        
        if self.dataset != 'cifar10':
            raise ValueError(f"Dataset {self.dataset} not supported")

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