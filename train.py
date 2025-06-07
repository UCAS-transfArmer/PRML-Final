import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import datetime

import wandb
from tqdm import tqdm
tqdm.pandas(ncols=80)#进度条宽度

from utils.arg_util import get_args
from dataloader.cifar_dataloader import get_dataloader

from utils import wandb_utils

from models import (
    LogisticRegression,
    ResNet,
    BoostingModel,
    VisionTransformer
)



def get_model(model_name, num_classes, image_dims, args):

    C, H, W = image_dims
    input_dim_flat = C * H * W # For models that take flattened input

    if model_name == 'logistic':
        model = LogisticRegression(input_dim=input_dim_flat, num_classes=num_classes)
    elif model_name == 'resnet':
        pass
    elif model_name == 'boosting':
        pass
    elif model_name == 'vit':
        
        model=VisionTransformer(
            image_size=args.image_size,
            patch_size=args.patch_size,
            num_classes=num_classes, 
            dim=args.dim, # 隐层维数
            depth=args.depth, # Transformer层数
            heads=args.heads, #注意力头数
            mlp_dim=args.mlp_dim, #MLP维数
            dropout=args.dropout,
            use_mlp_head=args.use_mlp_head   
        )
    else:
        raise ValueError(f"Model '{model_name}' not supported.")
    
    return model




def evaluate(model, dataloader, criterion, device, epoch_num=None):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # 添加验证集进度条
    pbar = tqdm(dataloader, desc='Evaluating')
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted_labels = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted_labels == labels).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0*correct_predictions/total_samples:.2f}%'
            })
    
    avg_loss = total_loss / total_samples
    accuracy = 100.0 * correct_predictions / total_samples
    
    epoch_str = f"Epoch {epoch_num} " if epoch_num is not None else ""
    print(f'{epoch_str}Test Set: Avg. Loss: {avg_loss:.4f}, Accuracy: {correct_predictions}/{total_samples} ({accuracy:.2f}%)')
    return avg_loss, accuracy




def get_scheduler(optimizer, args):
    """获取带预热的余弦学习率调度器"""
    # 计算总步数
    num_steps_per_epoch = 50000 // args.bs  # CIFAR-10 训练集大小
    total_steps = args.ep * num_steps_per_epoch
    warmup_steps = args.warmup_epochs * num_steps_per_epoch
    
    from torch.optim.lr_scheduler import OneCycleLR
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.tblr,          # 峰值学习率 3e-4
        total_steps=total_steps,
        pct_start=args.warmup_epochs / args.ep,  # 预热阶段比例
        div_factor=25,             # 初始学习率 = max_lr/25
        final_div_factor=1e4,      # 最终学习率 = 初始学习率/1e4
        three_phase=True,          # 使用三阶段策略
        anneal_strategy='cos'      # 余弦退火
    )
    
    return scheduler

def train(args):
    device = torch.device(args.device)
    
    # 打印设备信息
    print("\n设备配置:")
    print(f"- 指定设备: {args.device}")
    print(f"- 实际使用: {device}")
    if device.type == 'cuda':
        print(f"- GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"- CUDA版本: {torch.version.cuda}")
        print(f"- 可用显存: {torch.cuda.get_device_properties(0).total_memory/1024**2:.1f}MB")

    # 训练开始时启用CUDNN自动优化
    torch.backends.cudnn.benchmark = True

    # DataLoaders
    trainloader, testloader, num_classes, image_dims = get_dataloader(
        dataset_name=args.dataset, 
        batch_size=args.bs, 
        num_workers=args.num_workers, 
        data_root='./data', 
        for_vit=True,
        enhanced_augmentation=False
    )

    print(f"加载数据集: {args.dataset}, {num_classes}个类别, 图像维度: {image_dims}")
    
    # Model
    model = get_model(args.model, num_classes, image_dims, args)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params:,}")

    # Loss function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.tblr/25)  # 从较小的学习率开始
    scheduler = get_scheduler(optimizer, args)
    
    # 更新wandb配置
    wandb.config.update({
        "model_type": "ViT",
        "image_size": args.image_size,
        "patch_size": args.patch_size,
        "dim": args.dim,
        "depth": args.depth,
        "heads": args.heads,
        "mlp_dim": args.mlp_dim,
        "dropout": args.dropout,
        "batch_size": args.bs,
        "epochs": args.ep,
        "learning_rate": args.tblr,
        "optimizer": "AdamW",
        "device": str(device)
    })

    # 初始化训练状态
    global_step = 0
    best_acc = 0
    start_time = time.time()
    
    print(f"\n开始训练 {args.ep} 个epoch...")

    try:
        for epoch in range(args.ep):
            model.train()
            running_loss = 0.0
            epoch_start_time = time.time()
            
            # 优化进度条显示
            pbar = tqdm(trainloader, 
                       desc=f'Epoch {epoch+1}/{args.ep}',
                       leave=True,
                       ncols=100,
                       unit='batch')
            
            # 只在第一个epoch检查输入尺寸
            if epoch == 0:
                print(f"\nInput shape: {next(iter(trainloader))[0].shape}")
            
            for i, (inputs, labels) in enumerate(pbar):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()  # 只在这里更新学习率
                
                running_loss += loss.item()
                global_step += 1
                
                # 更新进度条显示
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{current_lr:.2e}',
                    'gpu_mem': f'{torch.cuda.memory_allocated()/1024**2:.0f}MB'
                }, refresh=True)
                
                # 记录到wandb
                if i % args.log_per_iter == 0:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/learning_rate": current_lr,
                        "system/gpu_memory": torch.cuda.memory_allocated()/1024**2,
                        "global_step": global_step
                    })
            
            # 计算平均损失
            avg_loss = running_loss / len(trainloader)
            
            # Epoch 结束总结
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"- Train Loss: {avg_loss:.4f}")
            print(f"- Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
            print(f"- Duration: {time.time() - epoch_start_time:.2f}s")
            
            # 每个epoch结束后记录验证指标
            val_loss, val_acc = evaluate(model, testloader, criterion, device, epoch_num=epoch + 1)
            
            # 记录到wandb，使用当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            wandb.log({
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "train/learning_rate": current_lr,
                "epoch": epoch
            })
            
            # 保存模型检查点到 wandb
            if (epoch + 1) % args.save_frequency == 0:
                checkpoint_path = f"checkpoint_epoch_{epoch+1}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, checkpoint_path)
                wandb.save(checkpoint_path)
            
            # 保存最佳模型
            if 'best_acc' not in locals() or val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(model, optimizer, epoch, args, is_best=True)  # 正确的参数顺序
            
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"警告: GPU内存不足!")
            torch.cuda.empty_cache()
        raise e
            
    except KeyboardInterrupt:
        print("\n训练被手动中断")
        save_checkpoint(model, optimizer, epoch, args, is_best=False)
        print("已保存检查点")
    finally:
        print(f"\n训练结束. 总用时: {datetime.timedelta(seconds=int(time.time()-start_time))}")
        wandb.finish()


def save_checkpoint(model, optimizer, epoch, args, is_best=False):
    """优化的检查点保存函数"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args)
    }
    
    # 保存当前检查点
    checkpoint_name = f'checkpoint_epoch_{epoch}.pth'
    checkpoint_path = os.path.join(args.save_path, checkpoint_name)
    torch.save(checkpoint, checkpoint_path)
    
    # 保存最佳模型
    if is_best:
        best_path = os.path.join(args.save_path, 'best_model.pth')
        torch.save(checkpoint, best_path)
        wandb.save(best_path)  # 只上传最佳模型到wandb


if __name__ == '__main__':
    args = get_args()
    
    print("Running with the following configuration:")
    for arg_name, value in vars(args).items():
        print(f"  {arg_name}: {value}")
    print("-" * 30)
   
    # Initialize Weights & Biases logging
    wandb_utils.initialize(
        args, 
        exp_name=args.exp_name, 
        project_name=args.project_name
    )
    
    train(args)