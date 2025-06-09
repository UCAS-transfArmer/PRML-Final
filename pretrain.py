import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import datetime
from tqdm.auto import tqdm

from utils.arg_util import get_args, Args 
from dataloader.dataloader import get_dataloader 
from utils.debug_data_util import get_debug_dataloaders 

from utils import wandb_utils
from utils.scheduler import create_scheduler 
from models import VisionTransformer 

from torch.cuda.amp import GradScaler, autocast

# --- 可选：分布式训练 ---
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP

def get_model_for_pretrain(args, num_classes, image_dims):
    """获取用于预训练的模型实例"""
    # 确保 image_dims 是 (C, H, W)
    # VisionTransformer 可能不直接使用 image_dims[0] 作为通道数，
    # 如果它内部硬编码为3或有其他方式确定。
    # 如果需要，可以修改 VisionTransformer 以接受 channels 参数。
    model = VisionTransformer(
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_classes=num_classes,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
        use_mlp_head=getattr(args, 'use_mlp_head', False)
        # channels=image_dims[0] # 如果模型支持，可以这样传递通道数
    )
    return model



def evaluate_pretrain(model, dataloader, criterion, device, epoch_num=None, use_amp=False):
    """在 ImageNet 验证集上评估预训练模型"""
    model.eval()
    total_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total_samples = 0
    
    pbar_desc = f'Evaluating Epoch {epoch_num}' if epoch_num is not None else 'Evaluating'
    pbar = tqdm(dataloader, desc=pbar_desc, leave=True, ncols=100)

    with torch.no_grad():
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            with autocast(enabled=use_amp): # 使用 autocast 进行评估
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0) # 乘以 batch size
            
            _, predicted_top1 = torch.max(outputs, 1)
            correct_top1 += (predicted_top1 == labels).sum().item()
            
            _, predicted_top5 = torch.topk(outputs, 5, dim=1)
            labels_reshaped = labels.view(-1, 1).expand_as(predicted_top5)
            correct_top5 += (predicted_top5 == labels_reshaped).sum().item()
            
            total_samples += labels.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'top1_acc': f'{100.0*correct_top1/total_samples:.2f}%' if total_samples > 0 else '0.00%',
                'top5_acc': f'{100.0*correct_top5/total_samples:.2f}%' if total_samples > 0 else '0.00%'
            })
    pbar.close() # 确保关闭进度条
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy_top1 = 100.0 * correct_top1 / total_samples if total_samples > 0 else 0.0
    accuracy_top5 = 100.0 * correct_top5 / total_samples if total_samples > 0 else 0.0
    
    epoch_str = f"Epoch {epoch_num} " if epoch_num is not None else ""
    print(f'{epoch_str}ImageNet Val Set: Avg. Loss: {avg_loss:.4f}, Top-1 Acc: {accuracy_top1:.2f}%, Top-5 Acc: {accuracy_top5:.2f}%')
    return avg_loss, accuracy_top1, accuracy_top5



def pretrain_imagenet(args):
    """在ImageNet上预训练模型的主函数"""
    device = torch.device(args.device)
    print(f"\n设备配置: 实际使用 {device}")
    if device.type == 'cuda' and torch.cuda.is_available(): # 增加检查cuda是否可用
        # 假设 args.device 是 "cuda" 或 "cuda:0" 等
        try:
            gpu_id = torch.cuda.current_device() if ':' not in args.device else int(args.device.split(':')[-1])
            print(f"- GPU型号: {torch.cuda.get_device_name(gpu_id)}")
        except Exception as e:
            print(f"获取GPU型号失败: {e}")

    torch.backends.cudnn.benchmark = True # 如果输入大小不变，可以加速训练
    use_amp = getattr(args, 'use_amp', False)
    if use_amp:
        print("使用混合精度训练 (AMP)")

    # --- 数据加载 ---
    if getattr(args, 'debug_use_fake_data', False): # 检查新的调试标志
        trainloader, testloader, num_classes, image_dims = get_debug_dataloaders(args)
    else:
        # 原始的真实数据加载逻辑
        print(f"为预训练加载数据集: {args.dataset}")
        trainloader, testloader, num_classes, image_dims = get_dataloader(
            dataset_name=args.dataset,
            batch_size=args.bs,
            num_workers=args.num_workers,
            data_root=args.data_root,
            for_vit=(args.model.lower()=='vit'), # 假设您的 get_dataloader 需要这个
            enhanced_augmentation=args.enhanced_augmentation,
            image_size=args.image_size,
            # 如果 get_dataloader 需要 ImageNet 子集参数，请确保传递它们
            ##################-----------------######################
        )
        # 数据加载后的验证检查
        if args.dataset.lower() == 'imagenet':
            expected_classes = 1000  # ImageNet-1K has 1000 classes
            if num_classes != expected_classes:
                print(f"警告: 数据加载器为ImageNet返回的类别数为 {num_classes}，期望为 {expected_classes}。")
                print(f"请检查dataloader实现或数据完整性。")
    
    # --- 模型 ---
    # image_dims 应该是 (C, H, W) 格式
    model_unwrapped = get_model_for_pretrain(args, num_classes, image_dims)

    # --- DataParallel 修改开始 ---
    if args.use_data_parallel and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"检测到 {torch.cuda.device_count()} 个 GPUs。启用 nn.DataParallel。")
        model = nn.DataParallel(model_unwrapped) # 包装模型
    else:
        model = model_unwrapped
        if args.use_data_parallel and (not torch.cuda.is_available() or torch.cuda.device_count() <= 1):
            print("请求了DataParallel，但条件不满足（无可用GPU/只有一个GPU）。在单个设备上运行。")
    # --- DataParallel 修改结束 ---

    model = model.to(device) # 将模型（或DataParallel包装器）移动到主设备

    total_params_unwrapped = sum(p.numel() for p in model_unwrapped.parameters() if p.requires_grad)
    print(f"模型参数量 (预训练, 未包装): {total_params_unwrapped:,}")


    # --- 损失函数和优化器 ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, # 使用 args.lr 作为初始学习率，调度器会处理 warmup_start_lr
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999) # AdamW 常用 beta 值
    )
    # 确保 create_scheduler 能够处理 warmup_start_lr
    scheduler = create_scheduler(optimizer, args) # 假设 args 包含 warmup_start_lr, warmup_epochs, min_lr 等
    scaler = GradScaler(enabled=use_amp)

    # --- W&B 初始化 ---
    # 确保 wandb_utils.initialize 和 wandb_utils.log 等函数存在且功能正确
    if getattr(args, 'use_wandb', True): # 假设 args 中有 use_wandb 字段，默认为 True
        try:
            wandb_utils.initialize(
                args, # 传递整个 args 对象
                exp_name=args.exp_name if args.exp_name else f"{args.model}_{args.dataset}_pretrain_debug_fake" if args.debug_use_fake_data else f"{args.model}_{args.dataset}_pretrain",
                project_name=args.project_name,
                model=model_unwrapped # watch 未包装的模型
            )
            print("WandB初始化成功。")
        except Exception as e:
            print(f"WandB初始化失败: {e}。将禁用WandB。")
            args.use_wandb = False # 出错则禁用
    else:
        print("WandB被禁用。")


    best_val_top1_acc = 0.0
    start_time = time.time()
    print(f"\n开始在 ImageNet 上预训练 {args.ep} 个 epochs...")

    try:
        for epoch in range(args.ep):
            model.train()
            running_loss_epoch = 0.0
            correct_train_epoch = 0
            total_train_epoch = 0
            epoch_start_time = time.time()

            print(f"\nEpoch {epoch+1}/{args.ep}")
            # 使用 unit='batch' 使进度条更清晰
            pbar = tqdm(trainloader, desc=f'Pre-training Epoch {epoch+1}', leave=False, ncols=100, unit='batch')

            for i, (inputs, labels) in enumerate(pbar):
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True) # set_to_none=True 可以提高性能

                with autocast(enabled=use_amp):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()

                if args.grad_clip_norm is not None and args.grad_clip_norm > 0:
                    scaler.unscale_(optimizer) 
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
                
                scaler.step(optimizer)
                scaler.update()

                running_loss_epoch += loss.item() * inputs.size(0) # 乘以 batch size
                _, predicted = torch.max(outputs, 1)
                total_train_epoch += labels.size(0)
                correct_train_epoch += (predicted == labels).sum().item()
                
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}', # 当前迭代的损失
                    'lr': f'{current_lr:.2e}',
                    'acc_train': f'{100.0 * correct_train_epoch / total_train_epoch:.2f}%' if total_train_epoch > 0 else '0.00%'
                })

                # WandB 日志记录迭代级别的信息
                if getattr(args, 'use_wandb', False) and wandb_utils.is_initialized() and (i+1) % args.log_per_iter == 0 :
                    iter_log_data = {
                        "pretrain/iter_loss": loss.item(),
                        "pretrain/lr": current_lr,
                        "pretrain/epoch_progress": epoch + (i+1)/len(trainloader), # 更精细的epoch进度
                    }
                    # 确保 wandb_utils.log 存在
                    wandb_utils.log(iter_log_data) # 移除了 step 参数，让 wandb 自动处理或在 log_epoch_metrics 中统一处理
            
            pbar.close() # 确保关闭进度条
            avg_train_loss = running_loss_epoch / total_train_epoch if total_train_epoch > 0 else 0.0
            avg_train_acc = 100.0 * correct_train_epoch / total_train_epoch if total_train_epoch > 0 else 0.0
            
            print(f"Epoch {epoch+1} Train: Loss={avg_train_loss:.4f}, Top-1 Acc={avg_train_acc:.2f}%, Time={time.time()-epoch_start_time:.2f}s")

            # --- 评估 ---
            val_loss, val_top1_acc, val_top5_acc = evaluate_pretrain(
                model, 
                testloader, 
                criterion, 
                device, 
                epoch_num=epoch + 1, 
                use_amp=use_amp
            )

            # WandB 日志记录 epoch 级别的信息
            if getattr(args, 'use_wandb', False) and wandb_utils.is_initialized():
                # 确保 wandb_utils.log_epoch_metrics 存在
                wandb_utils.log_epoch_metrics(
                    epoch + 1, 
                    avg_train_loss, 
                    val_loss, 
                    val_top1_acc, 
                    optimizer.param_groups[0]['lr'], # 当前 epoch 结束时的学习率
                    val_acc_top5=val_top5_acc,
                    train_acc_top1=avg_train_acc # 也记录训练集准确率
                )

            is_best = val_top1_acc > best_val_top1_acc
            if is_best:
                best_val_top1_acc = val_top1_acc
                print(f"新的最佳ImageNet Top-1验证准确率: {best_val_top1_acc:.2f}%")

            # 保存检查点
            if (epoch + 1) % args.save_frequency == 0 or is_best or (epoch + 1) == args.ep:
                state_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                # 确保 wandb_utils.save_checkpoint 存在
                wandb_utils.save_checkpoint(
                    state_to_save, 
                    optimizer.state_dict(), # 传递 optimizer.state_dict()
                    scheduler.state_dict(), # 传递 scheduler.state_dict()
                    epoch + 1, 
                    args, 
                    is_best=is_best,
                    checkpoint_name=f"pretrain_ckpt_epoch_{epoch+1}{'_best' if is_best else ''}.pth",
                    extra_state={'scaler_state_dict': scaler.state_dict()} if use_amp else None
                )
            
            scheduler.step() # 在每个 epoch 后更新学习率 (如果调度器是按epoch更新的话)

    except KeyboardInterrupt:
        print("\n预训练被用户中断。")
        if 'epoch' in locals() and model is not None and optimizer is not None and scheduler is not None and scaler is not None: # 检查变量是否存在
            state_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            wandb_utils.save_checkpoint(
                state_to_save, 
                optimizer.state_dict(), # 传递 optimizer.state_dict()
                scheduler.state_dict(), # 传递 scheduler.state_dict()
                epoch + 1 if 'epoch' in locals() else 0, 
                args, 
                is_best=False, # 中断时不是最佳
                checkpoint_name="pretrain_ckpt_interrupted.pth",
                extra_state={'scaler_state_dict': scaler.state_dict()} if use_amp else None
            )
            print("已保存中断时的检查点。")
    finally:
        print(f"\n预训练结束。总用时: {datetime.timedelta(seconds=int(time.time() - start_time))}")
        if getattr(args, 'use_wandb', False) and wandb_utils.is_initialized():
            wandb_utils.finish()


if __name__ == '__main__':
    args = get_args()
    
    print("使用以下配置运行ImageNet预训练:")
    for arg_name, value in sorted(vars(args).items()):
        print(f"  {arg_name}: {value}")
    print("-" * 30)

    # 参数验证 (可选，但推荐)
    try:
        temp_args_for_validation = Args(**vars(args)) # 现在 Args 是已定义的
        temp_args_for_validation.process_args()
        print("参数验证成功。")
    except ValueError as e:
        print(f"参数验证失败: {e}")
        exit(1)
    except Exception as e:
        print(f"解析或验证参数时发生其他错误: {e}")
        # exit(1) 

    pretrain_imagenet(args)