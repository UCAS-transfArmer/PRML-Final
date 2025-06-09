import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import datetime
from tqdm import tqdm

from utils.arg_util import get_args
from dataloader.dataloader import get_dataloader
from utils import wandb_utils  # 引入wandb_utils
from utils.scheduler import create_scheduler

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
    
    # 修改进度条配置，确保不会堆叠
    pbar = tqdm(dataloader, 
                desc='Evaluating', 
                leave=False,  # 改为False，使进度条完成后消失
                ncols=100)    # 保持宽度一致
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
        data_root=args.data_root, 
        for_vit=(args.model=='vit'),
        enhanced_augmentation=args.enhanced_augmentation,
        image_size=args.image_size,
        crop_padding=args.crop_padding if hasattr(args,'crop_padding') else 28
    )

    print(f"加载数据集: {args.dataset}, {num_classes}个类别, 图像维度: {image_dims}")
    
    # Model
    model = get_model(args.model, num_classes, image_dims, args)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params:,}")

    # Loss function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.warmup_start_lr,  # 从预热起始学习率开始
        weight_decay=0.05,        # ViT论文中使用的权重衰减
        betas=(0.9, 0.999)
    )
    scheduler = create_scheduler(optimizer, args)
    
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
            
            # 确保每个epoch只有一个进度条
            print(f"\nEpoch {epoch+1}/{args.ep}")  # 先打印当前epoch信息
            
            # 修改进度条配置
            pbar = tqdm(trainloader, 
                      desc='Training',  # 简化描述
                      leave=False,      # 完成后清除进度条
                      ncols=100,
                      unit='batch')
            
            # 只在第一个epoch检查输入尺寸
            if epoch == 0:
                print(f"Input shape: {next(iter(trainloader))[0].shape}")
            
            for i, (inputs, labels) in enumerate(pbar):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                global_step += 1
                
                # 更新进度条显示
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{current_lr:.2e}',
                    'gpu_mem': f'{torch.cuda.memory_allocated()/1024**2:.0f}MB'
                })  # 移除refresh=True，使用默认更新机制
            
            # 清除进度条
            pbar.close()
            
            # 计算平均损失
            avg_loss = running_loss / len(trainloader)
            
            # Epoch 结束总结
            print(f"Epoch {epoch + 1} Summary:")
            print(f"- Train Loss: {avg_loss:.4f}")
            print(f"- Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
            print(f"- Duration: {time.time() - epoch_start_time:.2f}s")
            
            # 每个epoch结束后记录验证指标
            val_loss, val_acc = evaluate(model, testloader, criterion, device, epoch_num=epoch + 1)
            
            # 使用wandb_utils记录epoch指标
            current_lr = optimizer.param_groups[0]['lr']
            wandb_utils.log_epoch_metrics(epoch, avg_loss, val_loss, val_acc, current_lr)
            
            # 保存模型检查点
            if (epoch + 1) % args.save_frequency == 0:
                wandb_utils.save_checkpoint(
                    model, optimizer, epoch, args, 
                    is_best=False,
                    checkpoint_name=f"checkpoint_epoch_{epoch+1}.pth"
                )
            
            # 保存最佳模型
            if 'best_acc' not in locals() or val_acc > best_acc:
                best_acc = val_acc
                wandb_utils.save_checkpoint(model, optimizer, epoch, args, is_best=True)
            
            # 更新学习率
            scheduler.step()
            
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"警告: GPU内存不足!")
            torch.cuda.empty_cache()
        raise e
            
    except KeyboardInterrupt:
        print("\n训练被手动中断")
        wandb_utils.save_checkpoint(model, optimizer, epoch, args, is_best=False)
        print("已保存检查点")
    finally:
        print(f"\n训练结束. 总用时: {datetime.timedelta(seconds=int(time.time()-start_time))}")
        wandb_utils.finish()


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