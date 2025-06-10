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
        # pass # Placeholder, ensure ResNet is correctly initialized if used
        raise NotImplementedError("ResNet model loading not fully implemented here.")
    elif model_name == 'boosting':
        # pass # Placeholder
        raise NotImplementedError("BoostingModel loading not fully implemented here.")
    elif model_name == 'vit':
        model=VisionTransformer(
            image_size=args.image_size,
            patch_size=args.patch_size,
            num_classes=num_classes,
            dim=args.dim,
            depth=args.depth,
            heads=args.heads,
            mlp_dim=args.mlp_dim,
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
    
    pbar_desc = f'Evaluating Epoch {epoch_num}' if epoch_num is not None else 'Evaluating'
    pbar = tqdm(dataloader, 
                desc=pbar_desc, 
                leave=False,
                ncols=100)
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted_labels = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted_labels == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0*correct_predictions/total_samples:.2f}%' if total_samples > 0 else '0.00%'
            })
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = 100.0 * correct_predictions / total_samples if total_samples > 0 else 0.0
    
    epoch_str = f"Epoch {epoch_num} " if epoch_num is not None else ""
    print(f'{epoch_str}Test Set: Avg. Loss: {avg_loss:.4f}, Accuracy: {correct_predictions}/{total_samples} ({accuracy:.2f}%)')
    return avg_loss, accuracy



def train(args):
    device = torch.device(args.device)
    
    print("\n设备配置:")
    print(f"- 指定设备: {args.device}")
    print(f"- 实际使用: {device}")
    if device.type == 'cuda':
        print(f"- GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"- CUDA版本: {torch.version.cuda}")
        print(f"- 可用显存: {torch.cuda.get_device_properties(0).total_memory/1024**2:.1f}MB")

    torch.backends.cudnn.benchmark = True

    trainloader, testloader, num_classes, image_dims = get_dataloader(
        dataset_name=args.dataset,
        batch_size=args.bs,
        num_workers=args.num_workers,
        data_root=args.data_root,
        for_vit=(args.model=='vit'),
        enhanced_augmentation=getattr(args, 'enhanced_augmentation', False),
        image_size=args.image_size,
    )

    print(f"加载数据集: {args.dataset}, {num_classes}个类别, 图像维度: {image_dims}")
    
    model = get_model(args.model, num_classes, image_dims, args)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=getattr(args, 'warmup_start_lr', args.lr), # Use warmup_start_lr if defined, else base lr
        weight_decay=getattr(args, 'weight_decay', 0.05), # Use defined weight_decay or default
        betas=(0.9, 0.999)
    )
    # Ensure create_scheduler is robust to missing scheduler-specific args if not used
    scheduler = create_scheduler(optimizer, args) 
    
    global_step = 0
    best_acc = 0.0 # Initialize best_acc
    start_time = time.time()
    
    print(f"\n开始训练 {args.ep} 个epoch...")

    # Check for grad_clip_norm attribute
    grad_clip_norm_val = getattr(args, 'grad_clip_norm', 0.0)
    if grad_clip_norm_val > 0:
        print(f"梯度裁剪已启用，最大范数: {grad_clip_norm_val}")


    try:
        for epoch in range(args.ep):
            model.train()
            running_loss = 0.0 # Changed from running_loss to running_loss_epoch for clarity
            epoch_start_time = time.time()
            
            print(f"\nEpoch {epoch+1}/{args.ep}")
            
            pbar = tqdm(trainloader, 
                      desc='Training',
                      leave=False,
                      ncols=100,
                      unit='batch')
            
            if epoch == 0 and len(trainloader) > 0:
                sample_inputs, _ = next(iter(trainloader))
                print(f"Input shape to model: {sample_inputs.shape}")
            
            for i, (inputs, labels) in enumerate(pbar):
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # --- 新增：梯度裁剪 ---
                if grad_clip_norm_val > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm_val)
                # --- 梯度裁剪结束 ---
                
                optimizer.step()
                
                running_loss += loss.item() # Accumulate loss.item() directly
                global_step += 1
                
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{current_lr:.2e}',
                    'gpu_mem': f'{torch.cuda.memory_allocated(device)/1024**2:.0f}MB' if device.type == 'cuda' else 'N/A'
                })
            
            pbar.close()
            
            # avg_loss = running_loss / len(trainloader) # This is average batch loss
            avg_epoch_loss = running_loss / len(trainloader) # More accurately, average loss per batch in epoch

            print(f"Epoch {epoch + 1} Summary:")
            print(f"- Train Avg. Batch Loss: {avg_epoch_loss:.4f}") # Clarified metric
            print(f"- Learning Rate (end of epoch): {optimizer.param_groups[0]['lr']:.2e}")
            print(f"- Duration: {time.time() - epoch_start_time:.2f}s")
            
            val_loss, val_acc = evaluate(model, testloader, criterion, device, epoch_num=epoch + 1)
            
            if getattr(args, 'use_wandb', False) and wandb_utils.is_initialized(): # Check if wandb is used and initialized
                current_lr_for_log = optimizer.param_groups[0]['lr']
                # Assuming log_epoch_metrics exists and works as intended
                wandb_utils.log_epoch_metrics(epoch, avg_epoch_loss, val_loss, val_acc, current_lr_for_log)
            
            save_frequency = getattr(args, 'save_frequency', 1) # Default save frequency to 1 if not set
            if (epoch + 1) % save_frequency == 0:
                # Simplified save_checkpoint call, ensure wandb_utils.save_checkpoint can handle this
                wandb_utils.save_checkpoint(
                    model.state_dict(), # Pass model state_dict
                    optimizer.state_dict(), # Pass optimizer state_dict
                    epoch, 
                    args, # Pass full args object
                    is_best=False,
                    checkpoint_name=f"checkpoint_epoch_{epoch+1}.pth"
                )
            
            if val_acc > best_acc:
                best_acc = val_acc
                print(f"New best validation accuracy: {val_acc:.2f}%")
                wandb_utils.save_checkpoint(
                    model.state_dict(), 
                    optimizer.state_dict(), 
                    epoch, 
                    args, 
                    is_best=True
                    # Default checkpoint name for best model or specify one
                )
            
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss) # Or val_acc, depending on strategy
                else:
                    scheduler.step() # For other schedulers like CosineAnnealingLR
            
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and device.type == 'cuda':
            print(f"警告: GPU内存不足!")
            torch.cuda.empty_cache()
        # Still raise the error to stop execution if it's critical
        raise e
            
    except KeyboardInterrupt:
        print("\n训练被手动中断")
        if 'epoch' in locals(): # Check if epoch is defined
            wandb_utils.save_checkpoint(
                model.state_dict(), optimizer.state_dict(), epoch, args, 
                is_best=False, checkpoint_name="checkpoint_interrupted.pth"
            )
            print("已保存中断检查点。")
        else:
            print("无法保存中断检查点，训练未开始或epoch未定义。")

    finally:
        total_training_time = time.time() - start_time
        print(f"\n训练结束. 总用时: {datetime.timedelta(seconds=int(total_training_time))}")
        if getattr(args, 'use_wandb', False) and wandb_utils.is_initialized():
            wandb_utils.finish()


if __name__ == '__main__':
    args = get_args()
    
    print("Running with the following configuration:")
    for arg_name, value in vars(args).items():
        print(f"  {arg_name}: {value}")
    print("-" * 30)
   
    if getattr(args, 'use_wandb', True):
        run_name = args.exp_name if args.exp_name else f"{args.model}_{args.dataset}_train"
        try:
            wandb_utils.initialize(
                args=args, 
                exp_name=run_name, 
                project_name=args.project_name,
                config=args, # Pass args as config
                # model=None # Model can be watched later if needed, after creation
            )
            print("WandB初始化成功。")
        except Exception as e:
            print(f"WandB初始化失败: {e}。将禁用WandB。")
            args.use_wandb = False 
    else:
        print("WandB被禁用。")
    
    train(args)