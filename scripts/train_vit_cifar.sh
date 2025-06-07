#!/bin/bash

echo "开始训练..."
python train.py \
    --model=vit \
    --dataset=cifar10 \
    --bs=256 \
    --ep=200 \
    --tblr=0.01 \
    --warmup_epochs=5 \
    --min_lr=3e-4 \
    --warmup_start_lr=1e-5 \
    --save_path=./ckpts \
    --exp_name=vit-cifar10-new \
    --image_size=32 \
    --patch_size=4 \
    --dim=384 \
    --depth=6 \
    --heads=6 \
    --mlp_dim=1536 \
    --dropout=0.1 \
    --num_workers=4 \
    --project_name=PRML-Final \
    --keep_n_checkpoints=2 \
    --save_frequency=40 \
    --log_per_iter=100 \
    --save_per_iter=1000 \
    --bs_base=256