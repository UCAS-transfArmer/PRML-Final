#!/bin/bash

echo "开始在ImageNet-1K上预训练ViT模型..."
python pretrain.py \
    --model=vit \
    --dataset=imagenet \
    --data_root=./data \
    --bs=256 \
    --ep=300 \
    --lr=3e-4 \
    --warmup_epochs=5 \
    --warmup_start_lr=1e-6 \
    --min_lr=1e-6 \
    --image_size=224 \
    --patch_size=16 \
    --dim=768 \
    --depth=12 \
    --heads=12 \
    --mlp_dim=3072 \
    --dropout=0.1 \
    --enhanced_augmentation \
    --weight_decay=0.05 \
    --grad_clip_norm=1.0 \
    --num_workers=8 \
    --project_name=PRML-Final \
    --exp_name=vit-base16-imagenet-pretrain \
    --save_path=./ckpts/imagenet \
    --keep_n_checkpoints=2 \
    --save_frequency=20 \
    --log_per_iter=100 \
    --use_data_parallel