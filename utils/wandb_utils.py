import wandb
import torch
from torchvision.utils import make_grid
import torch.distributed as dist
import os
import argparse
import hashlib
import math


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
        notes="从头开始训练ViT (使用优化后的学习率调度器)",  # 修改训练说明
        settings=wandb.Settings(
            start_method="thread",
            _disable_stats=True,
            sync_tensorboard=False
        )
    )
    
    # 添加学习率相关配置
    wandb.config.update({
        "lr_schedule": "one_cycle",
        "max_lr": args.tblr,
        "warmup_epochs": args.warmup_epochs,
        "min_lr": args.min_lr,
        "warmup_start_lr": args.warmup_start_lr
    })
    
    # 添加学习率图表配置
    wandb.define_metric("train/learning_rate", summary="min")
    wandb.define_metric("train/learning_rate", summary="max")
    wandb.define_metric("epoch")
    
    if model is not None:
        wandb.watch(model, log="gradients", log_freq=100)
    
    print("wandb日志初始化成功!")
    return wandb.run


def log(stats, step=None):
    if is_main_process():
        wandb.log({k: v for k, v in stats.items()}, step=step)


def log_image(name, sample, step=None):
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
    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            y_true=labels,
            preds=predictions,
            class_names=class_names
        )
    })

def log_sample_predictions(images, predictions, labels, class_names):
    """记录预测样本"""
    wandb.log({
        "sample_predictions": wandb.Image(
            images,
            caption=f"Pred: {class_names[predictions]}, True: {class_names[labels]}"
        )
    })