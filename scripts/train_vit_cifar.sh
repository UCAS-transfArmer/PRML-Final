#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "开始训练 ViT 模型..."

python train.py \
    --model=vit \
    --dataset=cifar10 \
    --data_root=./data \
    --bs=512 \
    --ep=200 \
    --lr=1e-4 \
    --warmup_epochs=10 \
    --warmup_start_lr=1e-5 \
    --min_lr=1e-6 \
    --image_size=224 \
    --patch_size=16 \
    --dim=768 \
    --depth=12 \
    --heads=12 \
    --mlp_dim=3072 \
    --dropout=0.1 \
    --weight_decay=0.1 \
    --enhanced_augmentation \
    --crop_padding=4 \
    --num_workers=16 \
    --project_name=PRML-Final \
    --exp_name=vit-base16-cifar10-rtxa6000-bs512-lr1e-4 \
    --save_path=./ckpts \
    --keep_n_checkpoints=2 \
    --save_frequency=5 \
    --log_per_iter=10 \
    --use_data_parallel \
    --use_amp

echo "训练完成！"
echo "WandB项目: PRML-Final"