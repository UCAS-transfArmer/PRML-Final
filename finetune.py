import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import datetime

from tqdm.auto import tqdm 

from utils.arg_util import get_args
from dataloader.dataloader import get_dataloader
from utils import wandb_utils # 使用wandb_utils
from utils.scheduler import create_scheduler
from models import VisionTransformer # 确保 VisionTransformer 模型定义正确

def load_pretrained_model(args, num_classes_finetune):
    """
    加载预训练模型，并根据微调任务调整模型（特别是分类头）。
    使用预训练模型的架构参数来构建模型。
    """
    print(f"\n加载预训练模型检查点: {args.pretrained_path}")
    
    if not os.path.exists(args.pretrained_path):
        raise FileNotFoundError(f"预训练模型文件不存在: {args.pretrained_path}")
    
    checkpoint = torch.load(args.pretrained_path, map_location='cpu')
    
    if 'args' not in checkpoint:
        raise KeyError("预训练检查点中未找到 'args'，无法确定原始模型架构。")
    pretrained_model_args = checkpoint['args']

    # 使用预训练时的架构参数，但 num_classes 和 dropout 可以使用当前微调的args
    model = VisionTransformer(
        image_size=pretrained_model_args.image_size,
        patch_size=pretrained_model_args.patch_size,
        num_classes=num_classes_finetune, # 微调任务的类别数
        dim=pretrained_model_args.dim,
        depth=pretrained_model_args.depth,
        heads=pretrained_model_args.heads,
        mlp_dim=pretrained_model_args.mlp_dim,
        dropout=args.dropout, # 使用微调时指定的dropout
        use_mlp_head=getattr(pretrained_model_args, 'use_mlp_head', False) # 与预训练模型一致
    )
    
    if 'model_state_dict' not in checkpoint:
        raise KeyError("预训练检查点中未找到 'model_state_dict'。")
    
    pretrained_dict = checkpoint['model_state_dict']
    cleaned_pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
    
    missing_keys, unexpected_keys = model.load_state_dict(cleaned_pretrained_dict, strict=False)
    
    print("预训练权重加载状态:")
    if missing_keys: print(f"  缺失的键 (通常是新的分类头): {missing_keys}")
    if unexpected_keys: print(f"  未预期的键: {unexpected_keys}")
    if not missing_keys and not unexpected_keys: print("  所有权重完美匹配。")
    elif all('head' in k for k in missing_keys) and not unexpected_keys:
        print("  成功加载骨干网络权重，分类头被重新初始化。")
    else:
        print("  请仔细检查缺失/未预期的键。")
        
    return model


def evaluate(model, dataloader, criterion, device, epoch_num=None):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc='Evaluating', leave=True, ncols=100)
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
    print(f'{epoch_str}Test Set: Avg. Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return avg_loss, accuracy


####FiineTUne主体####
def finetune(args):
    """微调模型的主函数"""
    device = torch.device(args.device)
    print(f"\n设备配置: 实际使用 {device}")
    if device.type == 'cuda': print(f"- GPU型号: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True

    trainloader, testloader, num_classes, _ = get_dataloader(
        dataset_name=args.dataset,
        batch_size=args.bs, 
        num_workers=args.num_workers,
        data_root=args.data_root, 
        for_vit=(args.model.lower()=='vit'),
        enhanced_augmentation=args.enhanced_augmentation,
        image_size=args.image_size
    )

    print(f"加载数据集: {args.dataset}, {num_classes}个类别, 图像将调整到: {args.image_size}x{args.image_size}")

    model = load_pretrained_model(args, num_classes)
    model = model.to(device)
    print(f"模型参数量 (微调): {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = nn.CrossEntropyLoss()
    
    head_keywords = ['head', 'mlp_head'] # 根据你的ViT模型分类头命名调整
    base_params = [p for n, p in model.named_parameters() if not any(keyword in n for keyword in head_keywords) and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if any(keyword in n for keyword in head_keywords) and p.requires_grad]

    if args.freeze_backbone:
        print("冻结骨干网络参数...")
        for param in base_params:
            param.requires_grad = False
        base_params = [p for p in base_params if p.requires_grad] # 更新列表

    optimizer_params = []
    if base_params:
        optimizer_params.append({'params': base_params, 'lr': args.warmup_start_lr, 'name': 'backbone'})
    if head_params:
        head_start_lr = args.warmup_start_lr * args.head_lr_multiplier
        optimizer_params.append({'params': head_params, 'lr': head_start_lr, 'name': 'head'})
    else: # 如果没有专门的头参数或者所有参数都被视为一体
        print("警告: 未找到特定头部参数或所有参数统一处理。")
        optimizer_params = [{'params': model.parameters(), 'lr': args.warmup_start_lr, 'name': 'all_params'}]


    if not any(pg['params'] for pg in optimizer_params):
        print("错误: 模型中没有可训练的参数。请检查 freeze_backbone 设置和模型结构。")
        return

    optimizer = optim.AdamW(optimizer_params, weight_decay=args.weight_decay)
    # 确保 args.lr (或 args.lr) 是 create_scheduler 的 max_lr (对应基础参数的峰值LR)
    # scheduler.py 中的 create_scheduler 应使用 args.lr 或 args.lr 作为基础参数的峰值学习率
    scheduler = create_scheduler(optimizer, args) 
    
    # 使用 wandb_utils 进行初始化
    run_name = args.exp_name if args.exp_name else f"{args.model}_{args.dataset}_finetune_lr{args.lr}_bs{args.bs}"

    wandb_utils.initialize(
        args, 
        exp_name=run_name, 
        project_name=args.project_name, 
        model=model
    )
    
    best_val_acc = 0.0
    start_time = time.time()
    print(f"\n开始微调 {args.ep} 个 epochs...")
    
    try:
        for epoch in range(args.ep):
            model.train()
            running_loss_epoch = 0.0
            correct_train_epoch = 0
            total_train_epoch = 0
            epoch_start_time = time.time()

            print(f"\nEpoch {epoch+1}/{args.ep}")
            pbar = tqdm(trainloader, desc=f'Fine-tuning Epoch {epoch+1}', leave=False, ncols=100, unit='batch')
            
            for i, (inputs, labels) in enumerate(pbar):
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()

                if args.grad_clip_norm is not None and args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
                optimizer.step()
                
                running_loss_epoch += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_train_epoch += labels.size(0)
                correct_train_epoch += (predicted == labels).sum().item()
                
                current_lrs = [group['lr'] for group in optimizer.param_groups]
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr_base': f'{current_lrs[0]:.2e}' if len(current_lrs)>0 else 'N/A',
                    'lr_head': f'{current_lrs[1]:.2e}' if len(current_lrs)>1 else 'N/A'
                })
                
                if (i+1) % args.log_per_iter == 0:
                    iter_log_data = {
                        "train/iter_loss": loss.item(),
                        "epoch": epoch + 1,
                    }
                    for idx, lr_val in enumerate(current_lrs):
                        group_name = optimizer.param_groups[idx].get('name', f'group_{idx}')
                        iter_log_data[f"train/lr_{group_name}"] = lr_val
                    wandb_utils.log(iter_log_data, step=epoch * len(trainloader) + i)
            
            pbar.close()
            avg_train_loss = running_loss_epoch / total_train_epoch if total_train_epoch > 0 else 0.0
            avg_train_acc = 100.0 * correct_train_epoch / total_train_epoch if total_train_epoch > 0 else 0.0
            
            print(f"Epoch {epoch+1} Train: Loss={avg_train_loss:.4f}, Acc={avg_train_acc:.2f}%, Time={time.time()-epoch_start_time:.2f}s")
            
            val_loss, val_acc = evaluate(model, testloader, criterion, device, epoch_num=epoch + 1)
            
            # 使用 wandb_utils.log_epoch_metrics
            # 注意：wandb_utils.log_epoch_metrics 可能需要调整以接受多个学习率或只记录基础学习率
            # 这里我们传递基础学习率，或者你可以修改 log_epoch_metrics
            base_lr_epoch_end = optimizer.param_groups[0]['lr'] if optimizer.param_groups else 0
            wandb_utils.log_epoch_metrics(epoch + 1, avg_train_loss, val_loss, val_acc, base_lr_epoch_end)
            
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                print(f"新的最佳验证准确率: {best_val_acc:.2f}%")
            
            if (epoch + 1) % args.save_frequency == 0 or is_best:
                # 使用 wandb_utils.save_checkpoint
                # wandb_utils.save_checkpoint 在其定义中不包含 scheduler 或 best_metric_val
                # 如果需要这些，需要修改 wandb_utils.save_checkpoint
                wandb_utils.save_checkpoint(
                    model, optimizer, epoch + 1, args, is_best=is_best,
                    checkpoint_name=f"finetuned_checkpoint_epoch_{epoch+1}{'_best' if is_best else ''}.pth"
                )
            
            scheduler.step()
            
    except KeyboardInterrupt:
        print("\n微调被用户中断。")
        if 'epoch' in locals() and model is not None and optimizer is not None:
             wandb_utils.save_checkpoint(
                model, optimizer, epoch + 1 if 'epoch' in locals() else 0, args, is_best=False, 
                checkpoint_name="finetuned_checkpoint_interrupted.pth"
            )
        print("已保存中断时的检查点。")
    finally:
        print(f"\n微调结束。总用时: {datetime.timedelta(seconds=int(time.time() - start_time))}")
        wandb_utils.finish()

################################
############主函数入口###########
################################

if __name__ == '__main__':
    args = get_args()
    
    # 为微调设置合理的默认参数 (如果命令行未提供)
    args.dataset = getattr(args, 'dataset', 'cifar10')
    args.model = getattr(args, 'model', 'vit')
    args.pretrained_path = getattr(args, 'pretrained_path', None)
    if not args.pretrained_path:
        raise ValueError("--pretrained_path 参数是必需的，用于指定预训练模型路径。")

    # 学习率和epoch通常在微调时较小
    args.lr = getattr(args, 'lr', 1e-4) # 基础参数的目标最大学习率
    args.ep = getattr(args, 'ep', 20) # 微调的总轮数
    args.bs = getattr(args, 'bs', 128) # 微调时的批量大小

    args.warmup_epochs = getattr(args, 'warmup_epochs', max(1, args.ep // 10)) # 例如预热10%的轮数
    args.warmup_start_lr = getattr(args, 'warmup_start_lr', 1e-7)
    args.min_lr = getattr(args, 'min_lr', 1e-6)
    args.weight_decay = getattr(args, 'weight_decay', 0.01) # 微调时权重衰减可小一些
    
    args.head_lr_multiplier = getattr(args, 'head_lr_multiplier', 10.0)
    args.freeze_backbone = getattr(args, 'freeze_backbone', False)
    args.grad_clip_norm = getattr(args, 'grad_clip_norm', 1.0) # 梯度裁剪范数

    # 日志和保存相关
    args.log_per_iter = getattr(args, 'log_per_iter', 100)
    args.save_frequency = getattr(args, 'save_frequency', 1) # 每多少个epoch保存一次
    args.save_path = getattr(args, 'save_path', './ckpts_finetune')

    print("使用以下配置运行微调:")
    for arg_name, value in sorted(vars(args).items()):
        print(f"  {arg_name}: {value}")
    print("-" * 30)
   
    finetune(args)