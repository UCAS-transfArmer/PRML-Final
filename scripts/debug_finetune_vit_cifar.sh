#!/bin/bash

echo "开始在 CIFAR-10 数据集上微调 ViT 模型（debug模式）..."

# 预训练模型路径 (指向伪检查点)

python finetune.py \
    --pretrained_path="./ckpts/debug/dummy_pretrained_vit_base16_224p16.pth" \
    --dataset="cifar10" \
    --data_root="./data" \
    --model="vit" \
    --bs=16 \
    --image_size=224 \
    --patch_size=16 \
    --crop_padding=28 \
    --dim=768 \
    --depth=12 \
    --heads=12 \
    --mlp_dim=3072 \
    --ep=5 \
    --lr=2e-5 \
    --warmup_epochs=1 \
    --warmup_start_lr=1e-7 \
    --min_lr=1e-6 \
    --weight_decay=0.01 \
    --head_lr_multiplier=10 \
    --grad_clip_norm=1.0 \
    --dropout=0.1 \
    --num_workers=2 \
    --device="cuda" \
    --project_name="PRML-Final" \
    --exp_name="debug_vit_cifar10_$(date +%Y%m%d-%H%M%S)" \
    --save_path="./ckpts_finetune_debug" \
    --keep_n_checkpoints=2 \
    --save_frequency=5 \
    --log_per_iter=20 \
    --enhanced_augmentation \
    --freeze_backbone \
    --use_amp \
    --use_mlp_head
   
     #--use_data_parallel \


echo "微调脚本执行完毕。"
