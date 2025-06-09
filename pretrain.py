import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import datetime
from tqdm.auto import tqdm

from utils.arg_util import get_args
from dataloader.dataloader import get_dataloader
from utils import wandb_utils
from utils.scheduler import create_scheduler
from models import VisionTransformer

# --- 混合精度 ---
from torch.cuda.amp import GradScaler, autocast

# --- 可选：分布式训练 ---
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP

def get_model_for_pretrain(args, num_classes, image_dims):
    """获取用于预训练的模型实例"""
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
            
            total_loss += loss.item() * inputs.size(0)
            
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
    if device.type == 'cuda': print(f"- GPU型号: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True
    use_amp = getattr(args, 'use_amp', False)
    if use_amp:
        print("使用混合精度训练 (AMP)")

    # --- 数据加载 ---
    print(f"为预训练加载数据集: {args.dataset} (应为 imagenet)")
    trainloader, testloader, num_classes, image_dims = get_dataloader(
        dataset_name=args.dataset,
        batch_size=args.bs,
        num_workers=args.num_workers,
        data_root=args.data_root,
        for_vit=(args.model.lower()=='vit'),
        enhanced_augmentation=args.enhanced_augmentation,
        image_size=args.image_size
    )
    if args.dataset.lower() == 'imagenet' and num_classes != 1000:
        print(f"警告: 数据加载器为ImageNet返回的类别数为 {num_classes}，期望为1000。请检查dataloader实现。")

    # --- 模型 ---
    model_unwrapped = get_model_for_pretrain(args, num_classes, image_dims)
    # model = model.to(device) # 原来的方式

    # --- DataParallel 修改开始 ---
    if args.use_data_parallel and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"检测到 {torch.cuda.device_count()} 个 GPUs。启用 nn.DataParallel。")
        # DataParallel 会将数据分发到所有可见的CUDA设备，并将模型复制到每个设备上。
        # 主设备（通常是args.device指定的，例如cuda:0）会收集结果。
        model = nn.DataParallel(model_unwrapped) # 包装模型
    else:
        model = model_unwrapped
    # --- DataParallel 修改结束 ---

    model = model.to(device) # 将模型（或DataParallel包装器）移动到主设备
                             # DataParallel 会处理其他GPU上的模型副本

    total_params_unwrapped = sum(p.numel() for p in model_unwrapped.parameters() if p.requires_grad)
    print(f"模型参数量 (预训练, 未包装): {total_params_unwrapped:,}")


    # --- 损失函数和优化器 ---
    criterion = nn.CrossEntropyLoss()
    # 如果使用DataParallel，优化器仍然作用于原始模型参数
    # 但通常我们直接将包装后的模型传入，PyTorch会处理
    optimizer = optim.AdamW(
        model.parameters(), # 直接使用 model.parameters() 即可
        lr=args.warmup_start_lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    scheduler = create_scheduler(optimizer, args)
    scaler = GradScaler(enabled=use_amp)

    # --- W&B 初始化 ---
    wandb_utils.initialize(
        args,
        exp_name=args.exp_name if args.exp_name else f"{args.model}_{args.dataset}_pretrain",
        project_name=args.project_name,
        model=model_unwrapped # 建议 watch 未包装的模型以获得更清晰的图结构
    )

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
            pbar = tqdm(trainloader, desc=f'Pre-training Epoch {epoch+1}', leave=False, ncols=100, unit='batch')

            for i, (inputs, labels) in enumerate(pbar):
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=use_amp): # 使用 autocast
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward() # 使用 scaler 缩放损失并反向传播

                if args.grad_clip_norm is not None and args.grad_clip_norm > 0:
                    scaler.unscale_(optimizer) # 在裁剪前 unscale 梯度
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
                
                scaler.step(optimizer) # scaler.step 会自动 unscale 梯度并调用 optimizer.step
                scaler.update() # 更新 scaler 的缩放因子

                running_loss_epoch += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_train_epoch += labels.size(0)
                correct_train_epoch += (predicted == labels).sum().item()
                
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{current_lr:.2e}',
                    'acc_train': f'{100.0 * correct_train_epoch / total_train_epoch:.2f}%' if total_train_epoch > 0 else '0.00%'
                })

                if (i+1) % args.log_per_iter == 0:
                    iter_log_data = {
                        "pretrain/iter_loss": loss.item(),
                        "pretrain/lr": current_lr,
                        "epoch": epoch + 1,
                    }
                    wandb_utils.log(iter_log_data, step=epoch * len(trainloader) + i)
            
            pbar.close()
            avg_train_loss = running_loss_epoch / total_train_epoch if total_train_epoch > 0 else 0.0
            avg_train_acc = 100.0 * correct_train_epoch / total_train_epoch if total_train_epoch > 0 else 0.0
            
            print(f"Epoch {epoch+1} Train: Loss={avg_train_loss:.4f}, Top-1 Acc={avg_train_acc:.2f}%, Time={time.time()-epoch_start_time:.2f}s")

            # --- 评估 ---
            val_loss, val_top1_acc, val_top5_acc = evaluate_pretrain(
                model, testloader, criterion, device, epoch_num=epoch + 1, use_amp=use_amp
            )

            wandb_utils.log_epoch_metrics(
                epoch + 1, 
                avg_train_loss, 
                val_loss, 
                val_top1_acc, 
                optimizer.param_groups[0]['lr'],
                val_acc_top5=val_top5_acc 
            )

            is_best = val_top1_acc > best_val_top1_acc
            if is_best:
                best_val_top1_acc = val_top1_acc
                print(f"新的最佳ImageNet Top-1验证准确率: {best_val_top1_acc:.2f}%")

            if (epoch + 1) % args.save_frequency == 0 or is_best:
                # 如果使用了 nn.DataParallel，模型状态字典的键会带有 'module.' 前缀
                # 保存时最好保存 model.module.state_dict() 以便更容易地在单GPU或不同GPU配置上加载
                state_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                wandb_utils.save_checkpoint(
                    state_to_save, # 直接传递状态字典
                    optimizer, epoch + 1, args, is_best=is_best,
                    checkpoint_name=f"pretrain_checkpoint_epoch_{epoch+1}{'_best' if is_best else ''}.pth",
                    # 如果需要保存scaler状态:
                    # extra_state={'scaler': scaler.state_dict()}
                )
            # ... (scheduler.step()) ...

    except KeyboardInterrupt:
        print("\n预训练被用户中断。")
        if 'epoch' in locals() and model is not None and optimizer is not None:
            state_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            wandb_utils.save_checkpoint(
                state_to_save, optimizer, epoch + 1 if 'epoch' in locals() else 0, args, is_best=False,
                checkpoint_name="pretrain_checkpoint_interrupted.pth"
                # extra_state={'scaler': scaler.state_dict()} if use_amp and 'scaler' in locals() else None
            )
            print("已保存中断时的检查点。")
    finally:
        print(f"\n预训练结束。总用时: {datetime.timedelta(seconds=int(time.time() - start_time))}")
        wandb_utils.finish()

if __name__ == '__main__':
    args = get_args()

    # --- 为 ImageNet 预训练设置合理的默认参数 ---
    args.dataset = getattr(args, 'dataset', 'imagenet')
    args.model = getattr(args, 'model', 'vit')
    
    args.ep = getattr(args, 'ep', 300)
    args.bs = getattr(args, 'bs', 256) 
    args.lr = getattr(args, 'lr', 1e-3) 
    
    args.warmup_epochs = getattr(args, 'warmup_epochs', 10)
    args.warmup_start_lr = getattr(args, 'warmup_start_lr', 1e-7)
    args.min_lr = getattr(args, 'min_lr', 1e-6)
    args.weight_decay = getattr(args, 'weight_decay', 0.05)
    
    args.image_size = getattr(args, 'image_size', 224)
    args.patch_size = getattr(args, 'patch_size', 16)
    args.dim = getattr(args, 'dim', 768)
    args.depth = getattr(args, 'depth', 12)
    args.heads = getattr(args, 'heads', 12)
    args.mlp_dim = getattr(args, 'mlp_dim', 3072)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.use_mlp_head = getattr(args, 'use_mlp_head', False)

    args.enhanced_augmentation = getattr(args, 'enhanced_augmentation', True)
    args.grad_clip_norm = getattr(args, 'grad_clip_norm', 1.0)

    args.log_per_iter = getattr(args, 'log_per_iter', 100)
    args.save_frequency = getattr(args, 'save_frequency', 20) # 修改了这里，与你之前的文件一致
    args.save_path = getattr(args, 'save_path', './ckpts_pretrain_imagenet')
    args.exp_name = getattr(args, 'exp_name', f"{args.model}_imagenet_pretrain_lr{args.lr}_bs{args.bs}")

    # --- AMP 参数 ---
    args.use_amp = getattr(args, 'use_amp', True) # 默认启用混合精度训练

    # --- DataParallel 参数 ---
    # 你可以在 arg_util.py 中添加这个参数，或者在这里用 getattr
    args.use_data_parallel = getattr(args, 'use_data_parallel', True) # 默认在多GPU时启用DataParallel

    print("使用以下配置运行ImageNet预训练:")
    for arg_name, value in sorted(vars(args).items()):
        print(f"  {arg_name}: {value}")
    print("-" * 30)

    pretrain_imagenet(args)