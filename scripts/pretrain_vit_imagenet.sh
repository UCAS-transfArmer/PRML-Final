#!/bin/bash

echo "开始在ImageNet-1K上预训练ViT模型..."
python train.py \
    --model=vit \
    --dataset=imagenet \
    --data_root=./data \
    --bs=256 \
    --ep=100 \
    --tblr=1e-3 \
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
    --num_workers=8 \
    --project_name=PRML-Final \
    --exp_name=vit-base16-imagenet-pretrain \
    --save_path=./ckpts/imagenet \
    --keep_n_checkpoints=2 \
    --save_frequency=5 \
    --log_per_iter=100 \
    --save_per_iter=5000