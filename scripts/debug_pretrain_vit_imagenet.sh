# filepath: /home/xushuwen/PRML-Final/scripts/debug_pretrain_vit_imagenet.sh
#!/bin/bash

echo "开始在ImageNet-1K子集上调试预训练ViT模型..."
python train.py \
    --model=vit \
    --dataset=imagenet \
    --data_root=./data_subset \
    --bs=16 \
    --ep=5 \
    --tblr=5e-4 \
    --warmup_epochs=1 \
    --warmup_start_lr=1e-7 \
    --min_lr=1e-7 \
    --image_size=224 \
    --patch_size=16 \
    --dim=768 \
    --depth=12 \
    --heads=12 \
    --mlp_dim=3072 \
    --dropout=0.1 \
    --enhanced_augmentation=false \
    --num_workers=4 \
    --project_name=PRML-Final-Debug \
    --exp_name=debug-vit-imagenet-subset-pretrain \
    --save_path=./ckpts_debug/imagenet_subset \
    --keep_n_checkpoints=1 \
    --save_frequency=1 \
    --log_per_iter=10 \
    --save_per_iter=50