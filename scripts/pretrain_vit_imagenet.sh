#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,6,7,8,9

echo "开始在ImageNet-1K上预训练ViT模型..."
python pretrain.py \
    --model=vit \
    --dataset=imagenet \
    --data_root=./data \
    --bs=1600 \
    --ep=300 \
    --lr=1.6e-3 \
    --warmup_epochs=6 \
    --warmup_start_lr=1e-6 \
    --min_lr=1e-5 \
    --image_size=224 \
    --patch_size=16 \
    --dim=768 \
    --depth=12 \
    --heads=12 \
    --mlp_dim=3072 \
    --dropout=0.0 \
    --enhanced_augmentation \
    --weight_decaykj0.05 \
    --grad_clip_norm=1.0 \
    --num_workers=16 \
    --project_name=PRML-Final \
    --exp_name=vit-base16-imagenet-pretrain-bs1600-lr1.6e-3 \
    --save_path=./ckpts/imagenet \
    --keep_n_checkpoints=3 \
    --save_frequency=10 \
    --log_per_iter=50 \
    --use_data_parallel \
    --use_amp
#--resume_checkpoint_path /path/to/your/checkpoint.pth
echo "脚本执行完毕。"