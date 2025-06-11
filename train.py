import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import datetime
from tqdm import tqdm
from torch.nn import DataParallel
# --- 修改：使用旧版 AMP API ---
from torch.cuda.amp import GradScaler, autocast 
import wandb #

from utils.arg_util import get_args
from dataloader.dataloader import get_dataloader
from utils import wandb_utils
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


def evaluate(model, dataloader, criterion, device, epoch_num=None, amp_enabled=False, device_type_for_amp='cuda'): # device_type_for_amp 在旧API中不再使用
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
            # --- 修改：使用旧版 autocast API ---
            with autocast(enabled=amp_enabled):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            # --- autocast结束 ---
            
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
        print(f"- GPU数量: {torch.cuda.device_count()}")
        if torch.cuda.device_count() > 0:
            print(f"- 当前GPU: {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")
            print(f"- CUDA版本: {torch.version.cuda}")

    torch.backends.cudnn.benchmark = True

    # --- 修改：混合精度训练设置，使用旧版 API ---
    use_amp_arg = getattr(args, 'use_amp', True) 
    amp_enabled = use_amp_arg and device.type == 'cuda' # AMP 主要用于 CUDA
    
    # GradScaler 初始化：使用旧版 API
    scaler = GradScaler(enabled=amp_enabled) 

    if amp_enabled:
        print(f"混合精度训练 (AMP) 已启用。")
    else:
        print("混合精度训练 (AMP) 未启用。")
    # --- 混合精度设置结束 ---

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

    if args.use_wandb and wandb_utils.is_initialized() and hasattr(wandb, 'watch'): # 确保 wandb 和 watch 可用
        wandb.watch(model, log="gradients", log_freq=100) # 在模型创建和移到设备后调用
    
    # --- 数据并行 (DP) 设置 ---
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        num_gpus_to_use=min(torch.cuda.device_count(),4) # 限制最多使用4个GPU

        if args.use_data_parallel: 
            if torch.cuda.device_count() >= num_gpus_to_use and num_gpus_to_use > 1 : 
                print(f"使用 {num_gpus_to_use} 个GPU进行数据并行训练。")
                device_ids = list(range(num_gpus_to_use)) 
                model = DataParallel(model, device_ids=device_ids)
            elif num_gpus_to_use <=1 and torch.cuda.device_count() > 0 :
                print(f"GPU数量不足或设置为1，将在单个GPU {device} 上运行。")
            else:
                print("未启用数据并行或无可用GPU。")
        else:
            print("数据并行被禁用 (use_data_parallel=False)。")
    # --- DP 设置结束 ---
        
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=getattr(args, 'warmup_start_lr', args.lr), 
        weight_decay=getattr(args, 'weight_decay', 0.05), 
        betas=(0.9, 0.999)
    )
    scheduler = create_scheduler(optimizer, args) 
    
    global_step = 0
    best_acc = 0.0 
    start_time = time.time()
    
    print(f"\n开始训练 {args.ep} 个epoch...")

    grad_clip_norm_val = getattr(args, 'grad_clip_norm', 0.0)
    if grad_clip_norm_val > 0:
        print(f"梯度裁剪已启用，最大范数: {grad_clip_norm_val}")

    try:
        for epoch in range(args.ep):
            model.train()
            running_loss = 0.0 
            epoch_start_time = time.time()
            
            print(f"\nEpoch {epoch+1}/{args.ep}")
            
            pbar = tqdm(trainloader, 
                      desc='Training',
                      leave=False,
                      ncols=100,
                      unit='batch')
            
            if epoch == 0 and len(trainloader) > 0:
                sample_inputs, _ = next(iter(trainloader))
                pbar.write(f"Input shape to model: {sample_inputs.shape}") 
            
            for i, (inputs, labels) in enumerate(pbar):
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                
                # --- 修改：使用旧版 autocast API ---
                with autocast(enabled=amp_enabled):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                # --- autocast结束 ---
                
                scaler.scale(loss).backward()
                
                grad_norm_to_log = 0.0 # Initialize

                # Unscale gradients first if AMP is enabled. This is crucial for both:
                # 1. Getting the correct (unscaled) gradient norm for logging.
                # 2. Ensuring clip_grad_norm_ (if used) operates on unscaled gradients.
                if amp_enabled:
                    scaler.unscale_(optimizer) 

                if grad_clip_norm_val > 0:
                    # clip_grad_norm_ will clip the gradients in-place.
                    # It returns the total norm of the gradients *before* clipping.
                    # Gradients are already unscaled if amp_enabled due to the call above.
                    grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm_val)
                    grad_norm_to_log = grad_norm_tensor.item()
                else:
                    # If not clipping, calculate the norm manually.
                    # Gradients are already unscaled if amp_enabled.
                    total_norm_sq = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2) 
                            total_norm_sq += param_norm.item() ** 2
                    grad_norm_to_log = total_norm_sq**0.5 if total_norm_sq > 0 else 0.0
                
                scaler.step(optimizer) # Applies optimizer step using unscaled gradients
                scaler.update()        # Updates the scaler for the next iteration
                
                running_loss += loss.item() 
                global_step += 1
                
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{current_lr:.2e}',
                    'gpu_mem': f'{torch.cuda.memory_allocated(device)/1024**2:.0f}MB' if device.type == 'cuda' else 'N/A'
                })

                # Log to WandB
                if args.use_wandb and wandb_utils.is_initialized() and \
                   (args.log_per_iter > 0 and global_step % args.log_per_iter == 0):
                    metrics_to_log_batch = {
                        "train/loss_batch": loss.item(),
                        "train/learning_rate_batch": current_lr,
                        "train/grad_norm": grad_norm_to_log
                    }
                    wandb_utils.log(metrics_to_log_batch, step=global_step)
            
            pbar.close()
            
            avg_epoch_loss = running_loss / len(trainloader) if len(trainloader) > 0 else 0.0

            print(f"Epoch {epoch + 1} Summary:")
            print(f"- Train Avg. Batch Loss: {avg_epoch_loss:.4f}") 
            print(f"- Learning Rate (end of epoch): {optimizer.param_groups[0]['lr']:.2e}")
            print(f"- Duration: {time.time() - epoch_start_time:.2f}s")
            
            # evaluate 函数中的 device_type_for_amp 参数在旧 API 中不再需要，但保留它不会导致错误
            val_loss, val_acc = evaluate(model, testloader, criterion, device, epoch_num=epoch + 1, amp_enabled=amp_enabled) 
            
            if getattr(args, 'use_wandb', False) and wandb_utils.is_initialized(): 
                current_lr_for_log = optimizer.param_groups[0]['lr']
                wandb_utils.log_epoch_metrics(epoch, avg_epoch_loss, val_loss, val_acc, current_lr_for_log)
            
            if val_acc > best_acc:
                best_acc = val_acc
                print(f"New best validation accuracy: {val_acc:.2f}%")
            
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss) 
                else:
                    scheduler.step() 
            
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and device.type == 'cuda':
            print(f"警告: GPU内存不足!")
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        raise e
            
    except KeyboardInterrupt:
        print("\n训练被手动中断")

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
   
    if getattr(args, 'use_wandb', True): # 默认启用WandB
        run_name = args.exp_name if args.exp_name else f"{args.model}_{args.dataset}_train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            wandb_utils.initialize(
                args=args, 
                exp_name=run_name, 
                project_name=args.project_name,
                config=vars(args), # 传递字典形式的配置
            )
            print("WandB初始化成功。") # 修改提示信息
            # 提示用户同步命令的 run_dir 通常在 wandb 初始化后可以从 wandb.run.dir 获取
            # 但这里我们先通用提示
        except Exception as e:
            print(f"WandB初始化失败: {e}。将禁用WandB。")
            args.use_wandb = False # 确保在失败时禁用
    else:
        print("WandB被禁用。")

    train(args)