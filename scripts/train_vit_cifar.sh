#!/bin/bash

echo "开始训练 ViT 模型..."

python train.py \
    --model=vit \
    --dataset=cifar10 \
    --data_root=./data \
    --bs=32 \
    --ep=200 \
    --lr=5e-5 \
    --warmup_epochs=20 \
    --warmup_start_lr=1e-6 \
    --min_lr=1e-7 \
    --image_size=224 \
    --patch_size=16 \
    --dim=768 \
    --depth=12 \
    --heads=12 \
    --mlp_dim=3072 \
    --dropout=0.1 \
    --enhanced_augmentation \
    --crop_padding=4 \
    --num_workers=2 \
    --project_name=PRML-Final \
    --exp_name=vit-base16-cifar10-rtx4060-bs32-lr5e5 \
    --save_path=./ckpts \
    --keep_n_checkpoints=2 \
    --save_frequency=25 \
    --log_per_iter=50
echo "训练完成！"
echo "检查点保存在: ./ckpts"
echo "WandB项目: PRML-Final"