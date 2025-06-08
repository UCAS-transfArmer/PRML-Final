import wandb
import torch
from torchvision.utils import make_grid
import torch.distributed as dist
import os
import argparse
import hashlib
import math
import datetime
import time


def is_main_process():
    # Default: single GPU
    return True
    # Multi-GPU training check
    # return dist.get_rank() == 0

def namespace_to_dict(namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
        for k, v in vars(namespace).items()
    }


def generate_run_id(exp_name):
    # https://stackoverflow.com/questions/16008670/how-to-hash-a-string-into-8-digits
    return str(int(hashlib.sha256(exp_name.encode('utf-8')).hexdigest(), 16) % 10 ** 8)


def initialize(args, exp_name, project_name, model=None):
    """初始化wandb日志记录"""
    print("初始化wandb日志...")
    
    group = "vit-cifar10"
    wandb.init(
        project=project_name,
        name=exp_name,
        config=vars(args),
        group=group,
        notes="开始训练ViT (使用优化后的学习率调度器)",
        settings=wandb.Settings(
            start_method="thread",
            _disable_stats=True,
            sync_tensorboard=False
        )
    )
    
    # 添加基本配置
    if args.model == 'vit':
        model_config = {
            "model_type": "ViT-Base/16",
            "model_variant": "Base/16",
            "image_size": args.image_size,
            "patch_size": args.patch_size,
            "dim": args.dim,
            "depth": args.depth,
            "heads": args.heads,
            "mlp_dim": args.mlp_dim,
            "dropout": args.dropout,
        }
        wandb.config.update(model_config)
    
    # 添加学习率相关配置
    lr_config = {
        "lr_schedule": "one_cycle",
        "max_lr": args.tblr,
        "warmup_epochs": args.warmup_epochs,
        "min_lr": args.min_lr,
        "warmup_start_lr": args.warmup_start_lr,
        "batch_size": args.bs,
        "epochs": args.ep,
        "optimizer": "AdamW",
        "device": args.device,
    }
    wandb.config.update(lr_config)
    
    # 添加学习率图表配置
    wandb.define_metric("train/learning_rate", summary="min")
    wandb.define_metric("train/learning_rate", summary="max")
    wandb.define_metric("epoch")
    
    if model is not None:
        wandb.watch(model, log="gradients", log_freq=100)
    
    print("wandb日志初始化成功!")
    return wandb.run


def log(stats, step=None):
    """统一的日志记录函数"""
    if is_main_process():
        wandb.log({k: v for k, v in stats.items()}, step=step)


def log_image(name, sample, step=None):
    """记录图像"""
    if is_main_process():
        sample = array2grid(sample)
        wandb.log({f"{name}": wandb.Image(sample), "train_step": step})


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x, nrow=nrow, normalize=True, value_range=(-1,1))
    x = x.mul(255).add_(0.5).clamp_(0,255).permute(1,2,0).to('cpu', torch.uint8).numpy()
    return x


def log_confusion_matrix(labels, predictions, class_names):
    """记录混淆矩阵"""
    if is_main_process():
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                y_true=labels,
                preds=predictions,
                class_names=class_names
            )
        })


def log_sample_predictions(images, predictions, labels, class_names):
    """记录预测样本"""
    if is_main_process():
        wandb.log({
            "sample_predictions": wandb.Image(
                images,
                caption=f"Pred: {class_names[predictions]}, True: {class_names[labels]}"
            )
        })


def log_training_batch(loss, current_lr, global_step, log_interval=100):
    """记录训练批次数据"""
    if is_main_process() and global_step % log_interval == 0:
        log({
            "train/loss": loss.item(),
            "train/learning_rate": current_lr,
            "system/gpu_memory": torch.cuda.memory_allocated()/1024**2,
            "global_step": global_step
        })


def log_epoch_metrics(epoch, train_loss, val_loss, val_acc, current_lr):
    """记录每个epoch的指标"""
    if is_main_process():
        log({
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "train/avg_loss": train_loss,
            "train/learning_rate": current_lr,
            "epoch": epoch
        })


def save_checkpoint(model, optimizer, epoch, args, is_best=False, checkpoint_name=None):
    """保存模型检查点"""
    if not is_main_process():
        return
        
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args)
    }
    
    # 确保保存目录存在
    os.makedirs(args.save_path, exist_ok=True)
    
    # 保存指定名称的检查点
    if checkpoint_name is None:
        checkpoint_name = f'checkpoint_epoch_{epoch}.pth'
    
    checkpoint_path = os.path.join(args.save_path, checkpoint_name)
    torch.save(checkpoint, checkpoint_path)
    
    # 在指定条件下上传到wandb
    if (epoch + 1) % args.save_frequency == 0:
        wandb.save(checkpoint_path)
    
    # 保存最佳模型
    if is_best:
        best_path = os.path.join(args.save_path, 'best_model.pth')
        torch.save(checkpoint, best_path)
        wandb.save(best_path)


def finish():
    """结束wandb记录"""
    if is_main_process():
        wandb.finish()