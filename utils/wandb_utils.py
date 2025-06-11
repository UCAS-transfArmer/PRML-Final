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
import re # Added for regex matching of checkpoint filenames

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


def initialize(args, exp_name, project_name, config,model=None):
    """初始化wandb日志记录"""
    print("初始化wandb日志...")
    
    group = "vit-cifar10"
    wandb.init(
        project=project_name,
        name=exp_name,
        config=config,
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
        "max_lr": args.lr,
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
    # wandb.define_metric("epoch") # Removed: epoch will no longer be a primary step metric

    # Define metrics for batch-level logging (primarily for train.py)
    wandb.define_metric("global_step") # Ensure global_step is defined
    wandb.define_metric("train/loss_batch", step_metric="global_step", summary="min")
    wandb.define_metric("train/learning_rate_batch", step_metric="global_step")
    wandb.define_metric("train/grad_norm", step_metric="global_step", summary="mean")

    # Define metrics for pretrain.py batch-level logging
    # wandb.define_metric("pretrain_step") # Removed: pretrain_step is replaced by global_step
    wandb.define_metric("pretrain/iter_loss", step_metric="global_step", summary="min")
    wandb.define_metric("pretrain/lr_batch", step_metric="global_step")
    wandb.define_metric("pretrain/grad_norm", step_metric="global_step", summary="mean")
    
    # Define metrics for epoch-level logging (used by both train.py and pretrain.py via log_epoch_metrics)
    # These will now use "global_step" as the step_metric.
    wandb.define_metric("val/epoch_loss", step_metric="global_step", summary="min")
    wandb.define_metric("val/epoch_top1_accuracy", step_metric="global_step", summary="max")
    wandb.define_metric("val/epoch_top5_accuracy", step_metric="global_step", summary="max") 
    wandb.define_metric("train/epoch_avg_loss", step_metric="global_step", summary="min")
    wandb.define_metric("train/epoch_learning_rate", step_metric="global_step") 
    wandb.define_metric("train/epoch_top1_accuracy", step_metric="global_step", summary="max") 
    
    if model is not None:
        wandb.watch(model, log="gradients", log_freq=100)
    
    print("wandb日志初始化成功!")
    return wandb.run


def is_initialized():
    """检查wandb是否已初始化"""
    return wandb.run is not None


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


def log_epoch_metrics(epoch, train_loss, val_loss, val_acc, current_lr, global_step_val, val_acc_top5=None, train_top1_acc=None): # Added global_step_val
    """记录每个epoch的指标"""
    if is_main_process():
        metrics_to_log = {
            "val/epoch_loss": val_loss, 
            "val/epoch_top1_accuracy": val_acc,
            "train/epoch_avg_loss": train_loss,
            "train/epoch_learning_rate": current_lr,
            "epoch": epoch # epoch is now logged as a metric, not as the x-axis step
        }
        if val_acc_top5 is not None:
            metrics_to_log["val/epoch_top5_accuracy"] = val_acc_top5
        if train_top1_acc is not None: 
            metrics_to_log["train/epoch_top1_accuracy"] = train_top1_acc
        
        log(metrics_to_log, step=global_step_val) # Pass global_step_val as the step


def save_checkpoint(model_state_dict, optimizer_state_dict, scheduler_state_dict, epoch, args,
                    is_best=False, checkpoint_name=None, extra_state=None):
    """保存模型检查点，包括模型状态、优化器状态、调度器状态、epoch和args。"""
    if not is_main_process():
        return
        
    # 确保保存目录存在
    os.makedirs(args.save_path, exist_ok=True)
    
    # 保存指定名称的检查点
    if checkpoint_name is None:
        checkpoint_name = f'checkpoint_epoch_{epoch}.pth'
    
    # 将 args (Tap 对象) 转换为可序列化的字典
    # 如果 args 是 Tap 对象，vars(args) 通常就足够了，因为它会返回其 __dict__
    # Tap 对象在设计上应该使其属性易于通过 vars() 获取
    args_to_save = args.as_dict() # <--- 修改：使用 as_dict() 方法

    checkpoint = {
        'epoch': epoch,
        'args': args_to_save,  # 修改：保存 args 的字典表示
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'scheduler_state_dict': scheduler_state_dict,
    }
    if extra_state:
        checkpoint.update(extra_state)
    
    checkpoint_path = os.path.join(args.save_path, checkpoint_name)
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}") # 打印保存信息
    
    # 在指定条件下上传到wandb
    # 移除 (epoch + 1) % args.save_frequency == 0 的判断，因为 pretrain.py 中已经有这个判断了
    # 只要调用 save_checkpoint，就应该尝试保存到 wandb (如果 wandb 启用)
    if wandb.run: # 检查 wandb.run 是否存在
        wandb.save(os.path.abspath(checkpoint_path), policy="live") # 使用 abspath 和 policy="live"
    
    # 保存最佳模型
    if is_best:
        best_path = os.path.join(args.save_path, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"Best model checkpoint saved to {best_path}") # 打印保存信息
        if wandb.run: # 检查 wandb.run 是否存在
            wandb.save(os.path.abspath(best_path), policy="live") # 使用 abspath 和 policy="live"

    # --- Logic to keep only N checkpoints ---
    if hasattr(args, 'keep_n_checkpoints') and args.keep_n_checkpoints is not None and args.keep_n_checkpoints > 0:
        # This pattern is specific to pretrain.py's naming convention
        # It matches "pretrain_ckpt_epoch_EPOCHNUMBER.pth" or "pretrain_ckpt_epoch_EPOCHNUMBER_best.pth"
        checkpoint_pattern = re.compile(r"pretrain_ckpt_epoch_(\d+)(_best)?\.pth")
        
        candidate_files = []
        try:
            candidate_files = os.listdir(args.save_path)
        except OSError as e:
            print(f"Error listing directory {args.save_path} for checkpoint cleanup: {e}")
            return # Stop cleanup if directory cannot be listed

        epoch_checkpoints = []
        for filename in candidate_files:
            match = checkpoint_pattern.fullmatch(filename)
            if match:
                epoch_num = int(match.group(1))
                epoch_checkpoints.append({
                    'path': os.path.join(args.save_path, filename),
                    'epoch': epoch_num,
                    'filename': filename
                })
        
        # Sort checkpoints by epoch number in ascending order (oldest first)
        epoch_checkpoints.sort(key=lambda x: x['epoch'])
        
        # If the number of checkpoints exceeds keep_n_checkpoints, remove the oldest ones
        if len(epoch_checkpoints) > args.keep_n_checkpoints:
            num_to_delete = len(epoch_checkpoints) - args.keep_n_checkpoints
            checkpoints_to_delete = epoch_checkpoints[:num_to_delete]
            
            for ckpt_info in checkpoints_to_delete:
                # The file 'best_model.pth' and '..._interrupted.pth' are not matched by the regex,
                # so they are naturally protected from this specific cleanup.
                try:
                    os.remove(ckpt_info['path'])
                    print(f"Removed old checkpoint (keep_n_checkpoints): {ckpt_info['filename']}")
                except OSError as e:
                    print(f"Error removing old checkpoint {ckpt_info['filename']}: {e}")

def finish():
    """结束wandb记录"""
    if is_main_process():
        wandb.finish()