# filepath: /home/xushuwen/PRML-Final/scripts/debug_finetune_vit_cifar.sh
#!/bin/bash

echo "开始在CIFAR-10上调试微调ViT模型 (基于ImageNet子集预训练)..."
python finetune.py \
    --model=vit \
    --dataset=cifar10 \
    --data_root=./data \
    --bs=32 \
    --ep=5 \
    --tblr=1e-4 \
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
    --crop_padding=28 \
    --num_workers=4 \
    --project_name=PRML-Final-Debug \
    --exp_name=debug-vit-cifar-finetune-from-subset \
    --save_path=./ckpts_debug/cifar_finetuned_from_subset \
    --pretrained_path=./ckpts_debug/imagenet_subset/best_model.pth \
    --keep_n_checkpoints=1 \
    --save_frequency=1 \
    --log_per_iter=10 \
    --save_per_iter=50