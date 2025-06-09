#!/bin/bash

echo "开始在ImageNet-1K上预训练ViT模型...(debug模式)"
python pretrain.py \
    --model=vit \
    --imagenet_use_subset \
    --imagenet_subset_num_classes=10 \
    --dataset=imagenet \
    --data_root=./data \
    --bs=8 \
    --ep=3 \
    --lr=1e-4 \
    --warmup_epochs=1 \
    --warmup_start_lr=1e-6 \
    --min_lr=1e-6 \
    --image_size=224 \
    --patch_size=16 \
    --dim=368 \
    --depth=6 \
    --heads=8 \
    --mlp_dim=1536 \
    --dropout=0.1 \
    --enhanced_augmentation \
    --weight_decay=0.05 \
    --grad_clip_norm=1.0 \
    --num_workers=0 \
    --project_name=PRML-Final \
    --exp_name=debug-pretrain-vit-imagenet-subset-bs8 \
    --save_path=./ckpts/debug \
    --keep_n_checkpoints=1 \
    --save_frequency=1 \
    --log_per_iter=5 \
    --use_data_parallel \
    --use_amp \
    --debug_use_fake_data