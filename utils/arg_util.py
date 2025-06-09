from tap import Tap
import torch
import os 
import argparse

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

def get_args():
    # 创建一个 Args 实例以获取其定义的默认值
    temp_args_for_defaults = Args()
    default_args = {
        # 从 temp_args_for_defaults 实例中获取所有在 Args 类中定义的字段及其值
        # 我们需要遍历 __annotations__ 来确保只获取在 Args 类中明确定义的字段
        # 而不是继承自 ArgumentParser 的其他属性。
        # 或者，更简单的方式是直接访问 temp_args_for_defaults 的属性，
        # 因为 Tap 会将这些字段设置为实例属性。
        # 然而，最安全的方式是直接使用 Tap 实例的 as_dict() 方法，
        # 但我们需要确保在调用 as_dict() 之前没有解析任何参数。
        # Tap 的 as_dict() 方法在未解析参数时，会返回定义的默认值。
    }
    # 遍历 Args 类的注解来获取字段名和默认值
    # 这种方法更精确地获取 Args 类中定义的字段
    args_instance_for_defaults = Args() # 创建一个干净的实例
    for field_name in args_instance_for_defaults.__annotations__:
        if hasattr(args_instance_for_defaults, field_name): # 确保字段存在于实例中
            default_args[field_name] = getattr(args_instance_for_defaults, field_name)


    parser = argparse.ArgumentParser(description="Vision Transformer Training Arguments")

    # Experiment specific arguments
    parser.add_argument('--project_name', type=str, default=default_args.get('project_name'), help='W&B project name')
    parser.add_argument('--exp_name', type=str, default=default_args.get('exp_name'), help='Experiment name for W&B')
    
    # Dataset and model selection
    parser.add_argument('--dataset', type=str, default=default_args.get('dataset'), choices=['cifar10', 'imagenet'], help='Dataset to use')
    parser.add_argument('--model', type=str, default=default_args.get('model'), choices=['logistic', 'boosting', 'resnet', 'vit'], help='Model architecture')
    parser.add_argument('--data_root', type=str, default=default_args.get('data_root'), help='Root directory of the dataset')
    
    # ViT specific arguments
    parser.add_argument('--image_size', type=int, default=default_args.get('image_size'), help='Input image size')
    parser.add_argument('--patch_size', type=int, default=default_args.get('patch_size'), help='Patch size for ViT')
    parser.add_argument('--dim', type=int, default=default_args.get('dim'), help='Embedding dimension for ViT')
    parser.add_argument('--depth', type=int, default=default_args.get('depth'), help='Number of transformer layers for ViT')
    parser.add_argument('--heads', type=int, default=default_args.get('heads'), help='Number of attention heads for ViT')
    parser.add_argument('--mlp_dim', type=int, default=default_args.get('mlp_dim'), help='MLP dimension in ViT')
    parser.add_argument('--dropout', type=float, default=default_args.get('dropout'), help='Dropout rate for ViT')
    # 确保为布尔值正确处理默认值
    parser.add_argument('--use_mlp_head', action='store_true', help='Use MLP head instead of linear head for ViT classification')
    if not default_args.get('use_mlp_head', False): # 如果默认是False，且命令行没提供，则不设置此参数的默认行为（即为False）
        # 如果 Args 中 use_mlp_head 默认为 True，则需要 parser.set_defaults(use_mlp_head=True)
        # 或者在 add_argument 中设置 default=True，但 action='store_true' 的 default 行为是 False
        pass # action='store_true' 默认就是 False 如果不出现
    if default_args.get('use_mlp_head'): # 如果Tap类中默认为True
        parser.set_defaults(use_mlp_head=True)


    # Training specific arguments
    parser.add_argument('--ep', type=int, default=default_args.get('ep'), help='Number of training epochs')
    parser.add_argument('--bs', type=int, default=default_args.get('bs'), help='Batch size')
    parser.add_argument('--lr', type=float, default=default_args.get('lr'), help='Base learning rate')
    parser.add_argument('--weight_decay', type=float, default=default_args.get('weight_decay'), help='Weight decay coefficient')

    # Learning rate scheduler
    parser.add_argument('--warmup_epochs', type=int, default=default_args.get('warmup_epochs'), help='Number of warmup epochs')
    parser.add_argument('--warmup_start_lr', type=float, default=default_args.get('warmup_start_lr'), help='Initial learning rate for warmup')
    parser.add_argument('--min_lr', type=float, default=default_args.get('min_lr'), help='Minimum learning rate for scheduler')

    # Data augmentation parameters
    parser.add_argument('--enhanced_augmentation', action='store_true', help='Use enhanced data augmentation')
    if default_args.get('enhanced_augmentation'):
        parser.set_defaults(enhanced_augmentation=True)
    parser.add_argument('--crop_padding', type=int, default=default_args.get('crop_padding'), help='Padding for random crop (relevant for CIFAR-10)')

    # ImageNet subset specific arguments for debugging
    parser.add_argument('--imagenet_use_subset', action='store_true', help='Use a subset of ImageNet for debugging')

    parser.add_argument('--imagenet_subset_num_classes', type=int, default=default_args.get('imagenet_subset_num_classes'), help='Number of classes for ImageNet subset')
    parser.add_argument('--imagenet_subset_samples_per_class', type=int, default=default_args.get('imagenet_subset_samples_per_class'), help='Samples per class for ImageNet subset (train)')
    parser.add_argument('--imagenet_subset_val_samples_per_class', type=int, default=default_args.get('imagenet_subset_val_samples_per_class'), help='Samples per class for ImageNet subset (validation)')

    # System and logging
    parser.add_argument('--num_workers', type=int, default=default_args.get('num_workers'), help='Number of data loading workers')
    parser.add_argument('--save_path', type=str, default=default_args.get('save_path'), help='Path to save checkpoints')
    parser.add_argument('--keep_n_checkpoints', type=int, default=default_args.get('keep_n_checkpoints'), help='Number of recent checkpoints to keep')
    parser.add_argument('--save_frequency', type=int, default=default_args.get('save_frequency'), help='Save checkpoint every N epochs')
    parser.add_argument('--log_per_iter', type=int, default=default_args.get('log_per_iter'), help='Log metrics every N iterations')
    parser.add_argument('--device', type=str, default=default_args.get('device'), help='Device to use (e.g., "cuda", "cpu")')
    parser.add_argument('--use_data_parallel', action='store_true', help='Use nn.DataParallel for multi-GPU training')
    if default_args.get('use_data_parallel'):
        parser.set_defaults(use_data_parallel=True)

    # Fine-tuning specific arguments
    parser.add_argument('--pretrained_path', type=str, default=default_args.get('pretrained_path'), help='Path to pretrained model checkpoint for fine-tuning')
    parser.add_argument('--head_lr_multiplier', type=float, default=default_args.get('head_lr_multiplier'), help='Multiplier for learning rate of the classification head during fine-tuning')
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze backbone weights during fine-tuning')
    if default_args.get('freeze_backbone'):
        parser.set_defaults(freeze_backbone=True)
    parser.add_argument('--grad_clip_norm', type=float, default=default_args.get('grad_clip_norm'), help='Max norm for gradient clipping')
    
    # Mixed precision training
    parser.add_argument('--use_amp', action='store_true', default=default_args.get('use_amp', False), help='Use Automatic Mixed Precision (AMP)')
    if default_args.get('use_amp'): # 确保Tap类中默认为True时，argparse也如此
        parser.set_defaults(use_amp=True)
    
    # Debugging
    parser.add_argument('--debug_skip_real_data', action='store_true', 
                        default=default_args.get('debug_skip_real_data', False), 
                        help='[调试用, 旧] 跳过真实数据加载，使用 pretrain.py 内的伪造数据逻辑')
    if default_args.get('debug_skip_real_data'):
        parser.set_defaults(debug_skip_real_data=True)

    parser.add_argument('--debug_use_fake_data', action='store_true',
                        default=default_args.get('debug_use_fake_data', False),
                        help='[调试用, 新] 使用 utils/debug_data_util.py 中的伪造数据进行调试')
    if default_args.get('debug_use_fake_data'):
        parser.set_defaults(debug_use_fake_data=True)
        
    parsed_args = parser.parse_args()
    return parsed_args

if __name__ == '__main__':
    # args_tap = Args().parse_args() # Example of using Tap directly
    # args_tap.process_args()
    # print("Parsed arguments with Tap:")
    # for arg_name, arg_val in args_tap.as_dict().items():
    #     print(f"  {arg_name}: {arg_val}")

    args_argparse = get_args()
    print("\nParsed arguments with argparse (get_args):")
    for arg_name, arg_val in vars(args_argparse).items():
        print(f"  {arg_name}: {arg_val}")
    
    # To validate args_argparse using the logic in Args.process_args:
    print("\nValidating argparse arguments using Args.process_args logic...")
    try:
        temp_args_for_validation = Args(**vars(args_argparse))
        temp_args_for_validation.process_args()
        print("Validation successful!")
    except ValueError as e:
        print(f"Validation failed: {e}")

    # print(f"\nRunning {args_argparse.model} model on {args_argparse.dataset} for {args_argparse.ep} epochs with base LR: {args_argparse.lr}")
    # print(f"Using device: {args_argparse.device}")
    # if args_argparse.pretrained_path:
    #     print(f"Attempting to load pretrained model from: {args_argparse.pretrained_path}")