from tap import Tap
import torch

class Args(Tap):
    # Experiment specific arguments
    project_name: str = 'prml-final'
    exp_name: str = 'default_experiment'  # Name of the experiment, used for logging and saving checkpoints.
    
    dataset: str = 'cifar10'  # Dataset to use. Currently only cifar10 is supported.
    
    # Model specific arguments
    model: str = 'resnet'  # Model architecture (choices: logistic, boosting, resnet, vit)
    
    # Training specific arguments
    ep: int = 400  # Number of epochs
    tblr: float = 1e-4  # Initial learning rate for bs = 1024
    bs: int = 1024  # Batch size
    bs_base: int = 1024  # Base batch size for scaling learning rate
    
    # Logging and Saving
    log_per_iter: int = 10  # Log training progress every N iterations
    save_per_iter: int = 1000  # Save model checkpoint every N iterations (0 to disable)
    save_path: str = 'ckpts'  # Path to save checkpoints
    
    # System specific arguments
    num_workers: int = 2  # Number of dataloader workers
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'  # Device to train on

    def process_args(self):
        if self.model not in ['logistic', 'boosting', 'resnet', 'vit']:
            raise ValueError(f"Model {self.model} not supported")
        
        if self.dataset != 'cifar10':
            raise ValueError(f"Dataset {self.dataset} not supported. Currently only cifar10 is supported.")

def get_args():
    args = Args().parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    print("Parsed arguments:")
    for arg_name, arg_val in vars(args).items():
        print(f"  {arg_name}: {arg_val}")
    
    print(f"\nRunning {args.model} model on {args.dataset} for {args.ep} epochs")
    print(f"Using device: {args.device}")