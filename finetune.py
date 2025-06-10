import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import datetime
from tqdm.auto import tqdm
from argparse import Namespace # 确保 Namespace 被导入

from utils.arg_util import get_args, Args # 复用arg_util
from dataloader.dataloader import get_dataloader # 复用dataloader
from utils import wandb_utils # 复用wandb_utils
from utils.scheduler import create_scheduler # 复用scheduler
from models.vit import VisionTransformer # 明确从models.vit导入

# --- 1. 模型加载和修改 ---
def load_and_prepare_model(args, num_classes_finetune):
    """
    加载预训练模型，并根据微调任务调整模型（特别是分类头）。
    """
    print(f"\\n加载预训练模型检查点: {args.pretrained_path}")
    if not os.path.exists(args.pretrained_path):
        raise FileNotFoundError(f"预训练模型文件不存在: {args.pretrained_path}")

    checkpoint = torch.load(args.pretrained_path, map_location='cpu',weights_only=False)
    
    #错误信息明确指出，从PyTorch2.6开始,torch.load函数的weights_only参数的默认值从 False 更改为了 True。
    #当 weights_only=True 时，torch.load 只会加载模型的权重（即 state_dict），并且出于安全考虑，它不允许反序列化（unpickle）任意的 Python 对象。

    if 'args' not in checkpoint:
        raise KeyError("预训练检查点中未找到 'args'，无法确定原始模型架构。")
    
    # 使用预训练时的架构参数来构建模型骨干
    # 但 num_classes 和 dropout 可以使用当前微调的 args
    # image_size 也应该从预训练的 args 中获取，以确保 patch embedding 兼容
    pretrained_model_args = checkpoint['args']
    
    # 确保 pretrained_model_args 是 Namespace 或类似对象，如果它是字典，需要转换
    if isinstance(pretrained_model_args, dict):
        pretrained_model_args = Namespace(**pretrained_model_args)


    print(f"  使用预训练模型的架构参数: image_size={pretrained_model_args.image_size}, "
          f"patch_size={pretrained_model_args.patch_size}, dim={pretrained_model_args.dim}, etc.")

    model = VisionTransformer(
        image_size=pretrained_model_args.image_size,        # 使用预训练时的 image_size
        patch_size=pretrained_model_args.patch_size,        # 使用预训练时的 patch_size
        num_classes=num_classes_finetune,                   # 微调任务的类别数
        dim=pretrained_model_args.dim,
        depth=pretrained_model_args.depth,
        heads=pretrained_model_args.heads,
        mlp_dim=pretrained_model_args.mlp_dim,
        dropout=args.dropout,                               # 使用微调时指定的dropout率
        use_mlp_head=args.use_mlp_head                      # <--- 修改这行参数适应微调 use_mlp_head
    )

    if 'model_state_dict' not in checkpoint:
        raise KeyError("预训练检查点中未找到 'model_state_dict'。")

    pretrained_dict = checkpoint['model_state_dict']

    cleaned_pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
    keys_to_remove_for_finetuning = []
    if num_classes_finetune != getattr(pretrained_model_args, 'num_classes', None): # 检查预训练时的类别数是否不同
        for key in cleaned_pretrained_dict.keys():
            if key.startswith('head.') or key.startswith('mlp_head.'): # 假设分类头相关的键以 'head.' 或 'mlp_head.' 开头
                keys_to_remove_for_finetuning.append(key)
    
    for key in keys_to_remove_for_finetuning:
        print(f"  从预训练权重中移除: {key}")
        del cleaned_pretrained_dict[key]

    # 加载权重，允许分类头不匹配
    missing_keys, unexpected_keys = model.load_state_dict(cleaned_pretrained_dict, strict=False)

    print("预训练权重加载状态:")
    if missing_keys: print(f"  缺失的键 (通常是新的分类头): {missing_keys}")
    if unexpected_keys: print(f"  未预期的键: {unexpected_keys}")
    
    if all('head' in k for k in missing_keys) and not unexpected_keys:
        print("  成功加载骨干网络权重，分类头被重新初始化。")
    elif not missing_keys and not unexpected_keys:
        print("  所有权重完美匹配 (可能预训练和微调任务类别数相同)。")
    else:
        print("  警告: 权重加载存在其他不匹配，请仔细检查。")
        
    return model, pretrained_model_args # 返回预训练的args，因为 image_size 可能需要

# --- 2. 评估函数 ---
def evaluate(model, dataloader, criterion, device, use_amp=False, epoch_num=None):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    pbar_desc = f'Evaluating Epoch {epoch_num}' if epoch_num is not None else 'Evaluating'
    pbar = tqdm(dataloader, desc=pbar_desc, leave=True, ncols=100)
    
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp): # 更新AMP API
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
    pbar.close()
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = 100.0 * correct_predictions / total_samples if total_samples > 0 else 0.0
    
    epoch_str = f"Epoch {epoch_num} " if epoch_num is not None else ""
    print(f'{epoch_str}Test Set: Avg. Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return avg_loss, accuracy

# --- 3. 微调主函数 ---
def finetune_main(args):
    """微调模型的主函数"""
    device = torch.device(args.device)
    print(f"\\n设备配置: 实际使用 {device}")
    if device.type == 'cuda' and torch.cuda.is_available():
        try:
            gpu_id = torch.cuda.current_device() if ':' not in args.device else int(args.device.split(':')[-1])
            print(f"- GPU型号: {torch.cuda.get_device_name(gpu_id)}")
        except Exception as e:
            print(f"获取GPU型号失败: {e}")
    torch.backends.cudnn.benchmark = True
    use_amp = getattr(args, 'use_amp', False) # 从args获取use_amp
    if use_amp: print("使用混合精度训练 (AMP)")

    # --- 数据加载 ---
    # 注意：微调时 image_size 应与预训练模型输入一致，dataloader 会处理 resize
    # get_dataloader 应该使用 args.image_size (由 finetune_vit_cifar.sh 设置，或从预训练模型推断)
    trainloader, testloader, num_classes_finetune, _ = get_dataloader(
        dataset_name=args.dataset,
        batch_size=args.bs,
        num_workers=args.num_workers,
        data_root=args.data_root,
        for_vit=(args.model.lower() == 'vit'),
        enhanced_augmentation=args.enhanced_augmentation,
        image_size=args.image_size, # 使用微调脚本中指定的 image_size
        crop_padding=args.crop_padding if args.dataset.lower() == 'cifar10' else 0 # crop_padding 主要用于CIFAR
    )
    print(f"加载数据集: {args.dataset}, {num_classes_finetune}个类别, 图像将调整到: {args.image_size}x{args.image_size}")

    # --- 模型加载和准备 ---
    model, pretrained_args = load_and_prepare_model(args, num_classes_finetune)
    
    # 如果微调脚本中的 image_size 与预训练的不一致，给出警告或调整
    if args.image_size != pretrained_args.image_size:
        print(f"警告: 微调脚本指定的 image_size ({args.image_size}) 与预训练模型使用的 image_size ({pretrained_args.image_size}) 不符。")
        print(f"将优先使用预训练模型的 image_size ({pretrained_args.image_size}) 进行数据加载和模型构建。")
        # 如果需要，这里可以强制 args.image_size = pretrained_args.image_size，并重新获取dataloader，
        # 但更好的做法是在 finetune_vit_cifar.sh 中就设置正确的 image_size，或者让 finetune.py 自动推断。
        # 当前 load_and_prepare_model 已经使用了 pretrained_args.image_size 构建模型。
        # dataloader 也应该使用这个尺寸。

    model_unwrapped = model # 在DP包装前保留未包装模型的引用
    if args.use_data_parallel and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"检测到 {torch.cuda.device_count()} 个 GPUs。启用 nn.DataParallel。")
        model = nn.DataParallel(model_unwrapped)
    else:
        if args.use_data_parallel:
             print("请求了DataParallel，但条件不满足。在单个设备上运行。")
    model = model.to(device)
    print(f"模型参数量 (微调, 可训练): {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- 损失函数、优化器、调度器 ---
    criterion = nn.CrossEntropyLoss()

    # 参数分组，为分类头设置不同的学习率
    head_keywords = ['head', 'mlp_head'] # 确保这些与你的ViT模型定义中的分类头层名称匹配
    
    # 冻结骨干网络
    if args.freeze_backbone:
        print("冻结骨干网络参数...")
        for name, param in model_unwrapped.named_parameters(): # 操作未包装的模型
            if not any(keyword in name for keyword in head_keywords):
                param.requires_grad = False
    
    # 分离基础参数和头部参数
    base_params = [p for name, p in model_unwrapped.named_parameters() if not any(keyword in name for keyword in head_keywords) and p.requires_grad]
    head_params = [p for name, p in model_unwrapped.named_parameters() if any(keyword in name for keyword in head_keywords) and p.requires_grad]

    optimizer_param_groups = []
    if base_params:
        optimizer_param_groups.append({'params': base_params, 'lr': args.lr, 'name': 'backbone'}) # 初始LR设为目标LR
        print(f"骨干网络参数组: {len(base_params)} tensors, 初始LR: {args.lr}")
    if head_params:
        optimizer_param_groups.append({'params': head_params, 'lr': args.lr * args.head_lr_multiplier, 'name': 'head'})
        print(f"头部参数组: {len(head_params)} tensors, 初始LR: {args.lr * args.head_lr_multiplier}")
    
    if not optimizer_param_groups: # 如果所有参数都被冻结了
        print("错误: 模型中没有可训练的参数。请检查 freeze_backbone 设置。")
        return

    optimizer = optim.AdamW(optimizer_param_groups, weight_decay=args.weight_decay)
    
    # create_scheduler 需要知道基础学习率是 args.lr
    # 调度器会根据 warmup_epochs 和 warmup_start_lr 调整初始学习率
    # temp_scheduler_args_dict = vars(args).copy() # 旧代码
    # temp_scheduler_args_dict['lr'] = args.lr    # 旧代码
    # scheduler_args = Args(**temp_scheduler_args_dict) # 旧代码，导致错误
    scheduler = create_scheduler(optimizer, args) # 修改：直接传递 args
    
    scaler = torch.amp.GradScaler(enabled=use_amp) # 移除 device_type，它会自动推断

    # --- W&B 初始化 ---
    if getattr(args, 'use_wandb', True): # 检查 use_wandb 属性
        run_name = args.exp_name if args.exp_name else f"{args.model}_{args.dataset}_finetune_lr{args.lr}_bs{args.bs}"
        try:
            # 确保 wandb_utils.initialize 接受这些参数
            wandb_utils.initialize(
                args=args,                       # 第一个参数：对应 initialize 中的 'args'
                exp_name=run_name,               # 第二个参数：对应 initialize 中的 'exp_name'
                project_name=args.project_name,  # 第三个参数：对应 initialize 中的 'project_name'
                config=args,                     # 第四个参数：对应 initialize 中的 'config' (使用 finetune.py 中的 args 对象作为配置)
                model=model_unwrapped            # 可选参数：对应 initialize 中的 'model'
            )
            print("WandB初始化成功。")
        except Exception as e:
            print(f"WandB初始化失败: {e}。将禁用WandB。")
            args.use_wandb = False # 在 args 对象上设置 use_wandb
    else:
        print("WandB被禁用。")

    # --- 训练循环 ---
    best_val_acc = 0.0
    start_time = time.time()
    print(f"\\n开始微调 {args.ep} 个 epochs...")

    try:
        for epoch in range(args.ep):
            model.train()
            running_loss_epoch = 0.0
            correct_train_epoch = 0
            total_train_epoch = 0
            epoch_start_time = time.time()

            print(f"\\nEpoch {epoch+1}/{args.ep}")
            pbar = tqdm(trainloader, desc=f'Fine-tuning Epoch {epoch+1}', leave=False, ncols=100, unit='batch')

            for i, (inputs, labels) in enumerate(pbar):
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type=device.type, enabled=use_amp): # 更新AMP API
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()

                # --- 新增：计算和准备梯度范数以供记录 ---
                grad_norm_to_log_this_iter = None
                # 首先，如果使用AMP，反缩放梯度
                if use_amp:
                    scaler.unscale_(optimizer) # 在计算范数或裁剪之前反缩放

                if getattr(args, 'use_wandb', False) and wandb_utils.is_initialized():
                    # 如果启用了梯度裁剪，clip_grad_norm_ 会返回裁剪前的范数
                    if args.grad_clip_norm is not None and args.grad_clip_norm > 0:
                        # 确保只对有梯度的可训练参数进行操作
                        params_to_clip = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
                        if params_to_clip:
                            # clip_grad_norm_ 会就地修改梯度
                            # 它返回的是裁剪前的总范数
                            grad_norm_val = torch.nn.utils.clip_grad_norm_(
                                params_to_clip,
                                max_norm=args.grad_clip_norm
                            )
                            grad_norm_to_log_this_iter = grad_norm_val.item()
                        else:
                            grad_norm_to_log_this_iter = 0.0 # 如果没有梯度（不太可能在训练中）
                    else:
                        # 如果没有梯度裁剪，手动计算范数
                        # 此时梯度已经反缩放（如果 use_amp 为 True）
                        current_total_norm = 0.0
                        for p in model.parameters():
                            if p.grad is not None and p.requires_grad:
                                param_norm = p.grad.data.norm(2)
                                current_total_norm += param_norm.item() ** 2
                        grad_norm_to_log_this_iter = (current_total_norm ** 0.5) if current_total_norm > 0 else 0.0
                else: # 如果不记录到wandb，但仍然启用了梯度裁剪
                    if args.grad_clip_norm is not None and args.grad_clip_norm > 0:
                        params_to_clip = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
                        if params_to_clip:
                            torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=args.grad_clip_norm)
                # --- 梯度范数计算结束 ---
                
                # scaler.step 会使用（现在已反缩放且可能已裁剪的）梯度
                scaler.step(optimizer)
                scaler.update()

                running_loss_epoch += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_train_epoch += labels.size(0)
                correct_train_epoch += (predicted == labels).sum().item()
                
                current_lrs_display = {pg.get('name', f'group{idx}'): f"{pg['lr']:.2e}" for idx, pg in enumerate(optimizer.param_groups)}
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    **current_lrs_display
                })

                if getattr(args, 'use_wandb', False) and wandb_utils.is_initialized():
                    iter_log_data = {"finetune/iter_loss": loss.item()}
                    for idx, pg in enumerate(optimizer.param_groups):
                        iter_log_data[f"finetune/lr_{pg.get('name', f'group{idx}')}"] = pg['lr']
                    
                    # 添加梯度范数到日志数据
                    if grad_norm_to_log_this_iter is not None:
                        iter_log_data["finetune/grad_norm"] = grad_norm_to_log_this_iter
                    
                    wandb_utils.log(iter_log_data) # wandb_utils.log 应该能处理 step

            pbar.close()
            avg_train_loss = running_loss_epoch / total_train_epoch if total_train_epoch > 0 else 0.0
            avg_train_acc = 100.0 * correct_train_epoch / total_train_epoch if total_train_epoch > 0 else 0.0
            print(f"Epoch {epoch+1} Train: Loss={avg_train_loss:.4f}, Acc={avg_train_acc:.2f}%, Time={time.time()-epoch_start_time:.2f}s")

            # --- 评估 ---
            val_loss, val_acc = evaluate(model, testloader, criterion, device, use_amp=use_amp, epoch_num=epoch + 1)

            if getattr(args, 'use_wandb', False) and wandb_utils.is_initialized():
                base_lr_epoch_end = optimizer.param_groups[0]['lr'] if optimizer.param_groups else 0
                # 确保 wandb_utils.log_epoch_metrics 接受这些参数
                wandb_utils.log_epoch_metrics(
                    epoch=epoch + 1, 
                    train_loss=avg_train_loss, 
                    val_loss=val_loss, 
                    val_acc=val_acc, 
                    current_lr=base_lr_epoch_end, # 修正: lr -> current_lr
                    train_top1_acc=avg_train_acc # 修正: train_acc -> train_top1_acc
                )

            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                print(f"新的最佳验证准确率: {best_val_acc:.2f}%")

            if (epoch + 1) % args.save_frequency == 0 or is_best or (epoch + 1) == args.ep:
                # 准备要保存的状态字典
                model_state_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                optimizer_state_to_save = optimizer.state_dict()
                scheduler_state_to_save = scheduler.state_dict()
                scaler_state_to_save = scaler.state_dict() if use_amp else None
                
                current_extra_state = {}
                if scaler_state_to_save is not None:
                    current_extra_state['scaler_state'] = scaler_state_to_save

                # 确保 wandb_utils.save_checkpoint 能处理这些参数
                wandb_utils.save_checkpoint(
                    model_state_dict=model_state_to_save,       # 修改: model_state -> model_state_dict
                    optimizer_state_dict=optimizer_state_to_save, # 修改: optimizer_state -> optimizer_state_dict
                    scheduler_state_dict=scheduler_state_to_save, # 修改: scheduler_state -> scheduler_state_dict
                    epoch=epoch + 1,
                    args=args, # 传递 args
                    is_best=is_best,
                    checkpoint_name=f"finetuned_ckpt_epoch_{epoch+1}{'_best' if is_best else ''}.pth",
                    extra_state=current_extra_state if current_extra_state else None 
                )
            
            scheduler.step() # 在每个 epoch 后更新学习率

    except KeyboardInterrupt:
        print("\\n微调被用户中断。")
        if 'epoch' in locals() and model is not None and optimizer is not None and scheduler is not None and scaler is not None:
            model_state_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            optimizer_state_to_save = optimizer.state_dict()
            scheduler_state_to_save = scheduler.state_dict()
            scaler_state_to_save = scaler.state_dict() if use_amp else None
            
            current_extra_state_interrupted = {}
            if scaler_state_to_save is not None:
                current_extra_state_interrupted['scaler_state'] = scaler_state_to_save

            wandb_utils.save_checkpoint(
                model_state_dict=model_state_to_save,        # 修改: model_state -> model_state_dict
                optimizer_state_dict=optimizer_state_to_save,  # 修改: optimizer_state -> optimizer_state_dict
                scheduler_state_dict=scheduler_state_to_save,  # 修改: scheduler_state -> scheduler_state_dict
                epoch=epoch + 1 if 'epoch' in locals() else 0, 
                args=args, 
                is_best=False,
                checkpoint_name="finetuned_ckpt_interrupted.pth",
                extra_state=current_extra_state_interrupted if current_extra_state_interrupted else None 
            )
            print("已保存中断时的检查点。")
    finally:
        total_runtime = time.time() - start_time
        print(f"\\n微调结束。总用时: {datetime.timedelta(seconds=int(total_runtime))}")
        if getattr(args, 'use_wandb', False) and wandb_utils.is_initialized():
            wandb_utils.finish()

# --- 4. 主函数入口 ---
if __name__ == '__main__':
    # 使用 get_args() 来获取命令行参数。
    # get_args() 内部使用 tap 库进行参数解析，并已调用 process_args() 进行验证。
    # 这允许 finetune_vit_cifar.sh 传递参数。
    args_cmd = get_args()
    
    # 验证 pretrained_path 是否提供
    if not args_cmd.pretrained_path:
        raise ValueError("--pretrained_path 参数是必需的，用于指定预训练模型路径。")

    # 可以在这里覆盖或设置一些微调特定的默认值（如果它们没有通过 .sh 脚本传递）
    # 但更好的做法是在 .sh 脚本中明确设置所有重要的微调参数。
    # 例如，确保微调时的学习率比预训练时小。
    # args_cmd.lr = getattr(args_cmd, 'lr', 1e-4) # 示例：如果脚本没传lr，则设为1e-4

    print("使用以下配置运行微调:")
    for arg_name, value in sorted(vars(args_cmd).items()):
        print(f"  {arg_name}: {value}")
    print("-" * 30)

    # 原先手动调用 Args.process_args() 的 try-except 代码块已被移除，
    # 因为 get_args() 现在负责处理参数的完整解析和验证流程。
    # 如果在 get_args() 内部（包括 tap 解析或 process_args 调用时）发生错误，
    # 程序通常会直接退出并显示相关错误信息。

    finetune_main(args_cmd)