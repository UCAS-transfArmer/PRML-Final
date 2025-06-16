#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "开始在 CIFAR-10 数据集上微调 ViT 模型..."

# 确保保存路径的父目录存在 (finetune.py 内部会根据 exp_name 创建最终的 save_path)

python finetune.py \
    --pretrained_path="./ckpts/imagenet/best_model.pth" \
    --dataset="cifar10" \
    --data_root="./data" \
    --model="vit" \
    --image_size=224 \
    --bs=512 \
    --ep=60 \
    --lr=2e-4 \
    --warmup_epochs=3 \
    --warmup_start_lr=1e-6 \
    --min_lr=1e-6 \
    --weight_decay=0.05 \
    --head_lr_multiplier=10 \
    --grad_clip_norm=1.0 \
    --dropout=0.1 \
    --num_workers=16 \
    --device="cuda" \
    --label_smoothing=0.1 \
    --project_name="PRML-Final" \
    --exp_name="vit_cifar10_ft_from_imagenet_lr2e-4_bs512" \
    --save_path="./ckpts/finetune" \
    --save_frequency=10 \
    --log_per_iter=20 \
    --enhanced_augmentation \
    --crop_padding=28 \
    --use_amp \
    --freeze_backbone \
    --use_data_parallel

echo "微调脚本执行完毕。"