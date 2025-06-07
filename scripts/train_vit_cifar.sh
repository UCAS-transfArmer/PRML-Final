#!/bin/bash

echo "开始训练..."
python train.py \
    --model=vit \
    --dataset=cifar10 \
    --bs=1024 \
    --ep=100 \
    --tblr=1e-2 \
    --warmup_epochs=10 \
    --warmup_start_lr=1e-5 \
    --min_lr=1e-5 \
    --image_size=224 \
    --patch_size=16 \
    --dim=768 \
    --depth=12 \
    --heads=12 \
    --mlp_dim=3072 \
    --dropout=0.1 \
    --num_workers=4 \
    --project_name=PRML-Final \
    --exp_name=vit-base16-cifar \
    --save_path=./ckpts \
    --keep_n_checkpoints=2 \
    --save_frequency=40 \
    --log_per_iter=100 \
    --save_per_iter=1000 \