import math
import torch
from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineScheduler(_LRScheduler):
    """
    带预热的余弦学习率调度器
    参考 ViT 论文实现：预热 + 余弦衰减
    """
    def __init__(
        self,
        optimizer,
        warmup_epochs,
        max_epochs,
        warmup_start_lr=1e-6,
        max_lr=3e-4,
        min_lr=1e-6,
        last_epoch=-1
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.last_epoch = last_epoch

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """计算当前epoch的学习率"""
        epoch = self.last_epoch + 1
        
        # 预热阶段：线性增加
        if epoch < self.warmup_epochs:
            alpha = epoch / self.warmup_epochs
            return [self.warmup_start_lr + (self.max_lr - self.warmup_start_lr) * alpha for _ in self.base_lrs]
            
        # 余弦衰减阶段
        progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        
        # 计算当前学习率
        current_lr = self.min_lr + (self.max_lr - self.min_lr) * cosine_decay
        return [current_lr for _ in self.base_lrs]

def create_scheduler(optimizer, args):
    """
    创建学习率调度器的工厂函数
    """
    return WarmupCosineScheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.ep,
        warmup_start_lr=args.warmup_start_lr,
        max_lr=args.tblr,
        min_lr=args.min_lr
    )