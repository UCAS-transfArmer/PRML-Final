#!/bin/bash

echo "开始训练 ViT 模型..."
python train.py \
    --model=vit \
    --dataset=cifar10 \
    --data_root=./data \
    --bs=128 \
    --ep=100 \
    --tblr=3e-4 \
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
    --enhanced_augmentation=true \
    --crop_padding=28 \
    --num_workers=4 \
    --project_name=PRML-Final \
    --exp_name=vit-base16-cifar-lr3e-4-bs128 \
    --save_path=./ckpts \
    --keep_n_checkpoints=2 \
    --save_frequency=50 \
    --log_per_iter=100