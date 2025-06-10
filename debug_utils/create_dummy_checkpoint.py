import torch
import os
from argparse import Namespace
from models.vit import VisionTransformer # 确保可以从 models.vit 导入 VisionTransformer

def create_dummy_vit_checkpoint(save_path_full):
    """
    创建一个伪的ViT预训练模型检查点，用于调试。
    """
    # 更新为 ViT-Base/16 (224x224 输入, 16x16 patch) 的配置
    dummy_args = Namespace(
        image_size=224,        # 目标图像大小
        patch_size=16,         # 目标 Patch 大小
        dim=768,               # ViT-Base 隐藏层维度
        depth=12,              # ViT-Base Transformer层数
        heads=12,              # ViT-Base 注意力头数
        mlp_dim=3072,          # ViT-Base MLP维度 (4 * dim)
        dropout=0.0,           # 预训练时通常 dropout 为 0 或较小值
        use_mlp_head=False,    # 通常预训练时使用线性头
        model='vit',           # 模型类型
        dataset='dummy_imagenet_pretrain', # 伪预训练时的数据集名称
        # 根据需要添加 VisionTransformer 或 load_and_prepare_model 可能期望从 checkpoint['args'] 中获取的其他参数
    )

    # 使用伪配置初始化模型
    # 此处 num_classes 可以是 ImageNet 的类别数 (1000) 或其他占位符
    dummy_num_classes_pretrain = 1000 
    model = VisionTransformer(
        image_size=dummy_args.image_size,
        patch_size=dummy_args.patch_size,
        num_classes=dummy_num_classes_pretrain,
        dim=dummy_args.dim,
        depth=dummy_args.depth,
        heads=dummy_args.heads,
        mlp_dim=dummy_args.mlp_dim,
        dropout=dummy_args.dropout, # 实际模型构建时，finetune.py 会用微调的dropout
        use_mlp_head=dummy_args.use_mlp_head
    )
    model.eval() # 设置为评估模式

    # 创建检查点字典
    checkpoint = {
        'args': dummy_args,             # 保存Namespace对象
        'model_state_dict': model.state_dict(),
        'epoch': -1,                    # 表示这是一个伪检查点
        'optimizer_state_dict': None,   # 微调时通常不需要
        'scaler_state_dict': None,      # 微调时通常不需要
        # 可以添加其他预训练脚本会保存的元信息，如 'best_val_acc' 等，但对于调试不是必需的
    }

    # 确保保存目录存在
    save_dir = os.path.dirname(save_path_full)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"创建目录: {save_dir}")

    # 保存检查点
    torch.save(checkpoint, save_path_full)
    print(f"伪ViT检查点已保存到: {save_path_full}")
    print(f"  模型配置: image_size={dummy_args.image_size}, patch_size={dummy_args.patch_size}, "
          f"dim={dummy_args.dim}, depth={dummy_args.depth}, heads={dummy_args.heads}")

if __name__ == "__main__":
    # 定义伪检查点的保存路径和文件名
    # 更新文件名以反映新的配置
    dummy_checkpoint_file = "./ckpts/debug/dummy_pretrained_vit_base16_224p16.pth"
    create_dummy_vit_checkpoint(save_path_full=dummy_checkpoint_file)