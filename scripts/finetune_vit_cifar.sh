#!/bin/bash

echo "开始在 CIFAR-10 数据集上微调 ViT 模型..."


# 确保保存路径的父目录存在 (finetune.py 内部会根据 exp_name 创建最终的 save_path)
# mkdir -p ./ckpts_finetune 

python finetune.py \
    --pretrained_path="./ckpts/imagenet/vit-base16-imagenet-pretrain-bs512-original/best_model.pth" \
    --dataset="cifar10" \
    --data_root="./data" \
    --model="vit" \
    --image_size=224 \
    --bs=64 \
    --ep=50 \
    --lr=2e-5 \
    --warmup_epochs=5 \
    --warmup_start_lr=1e-7 \
    --min_lr=1e-6 \
    --weight_decay=0.01 \
    --head_lr_multiplier=10 \
    --grad_clip_norm=1.0 \
    --dropout=0.1 \
    --num_workers=4 \
    --device="cuda" \
    --project_name="PRML-Final" \
    --exp_name="vit_cifar10_ft_from_imagenet_lr2e-5_bs64" \
    --save_path="./ckpts_finetune" \
    --save_frequency=5 \
    --log_per_iter=20 \
    --enhanced_augmentation \
    --crop_padding=28 \
    --use_amp \
    --use_data_parallel

echo "微调脚本执行完毕。"