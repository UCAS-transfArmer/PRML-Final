#!/bin/bash

# --- 用户需要配置的参数 ---
# **重要**: 指定你的预训练模型检查点路径
PRETRAINED_MODEL_PATH="./ckpts/vit-base16-cifar-lr3e-4-bs128/best_model.pth" # 示例路径，请修改为你的实际路径

# 微调任务相关参数
DATASET_NAME="cifar10"
DATA_ROOT="./data"
NUM_CLASSES_FINETUNE=10 # CIFAR-10 有10个类别

# ViT 模型架构参数 (通常应与预训练模型一致，但 image_size 可能需要在 finetune.py 中从预训练 args 获取)
# finetune.py 中的 load_pretrained_model 应该会从检查点中读取大部分架构参数
IMAGE_SIZE=224 # ViT 输入图像尺寸 (CIFAR10图像会被调整到这个尺寸)

# 微调训练参数
BATCH_SIZE=64
EPOCHS=30           # 微调轮数可以少一些
BASE_LR=5e-5        # 微调时基础学习率通常较小 (例如 1e-4, 5e-5, 2e-5)
WARMUP_EPOCHS=3     # 预热轮数 (例如总轮数的10%)
WARMUP_START_LR=1e-7
MIN_LR=1e-6
WEIGHT_DECAY=0.01
HEAD_LR_MULTIPLIER=10 # 分类头学习率是基础学习率的多少倍
FREEZE_BACKBONE="False" # 是否冻结骨干网络 ("True" 或 "False")
GRAD_CLIP_NORM=1.0    # 梯度裁剪范数 (0 表示不裁剪)
DROPOUT_RATE=0.1      # Dropout率

# 其他设置
NUM_WORKERS=4
DEVICE="cuda" # 或者 "cpu"

# W&B 和日志/保存设置
PROJECT_NAME="PRML-Final-Finetune"
EXP_NAME="vit_cifar10_finetune_lr${BASE_LR}_bs${BATCH_SIZE}"
SAVE_PATH="./ckpts_finetune/${EXP_NAME}"
SAVE_FREQUENCY=5
LOG_PER_ITER=50

# --- 执行微调脚本 ---
echo "开始微调 ViT 模型在 ${DATASET_NAME} 数据集上..."
echo "预训练模型路径: ${PRETRAINED_MODEL_PATH}"

# 确保保存路径存在
mkdir -p ${SAVE_PATH}

python finetune.py \
    --pretrained_path="${PRETRAINED_MODEL_PATH}" \
    --dataset="${DATASET_NAME}" \
    --data_root="${DATA_ROOT}" \
    --model="vit" \
    --image_size=${IMAGE_SIZE} \
    --bs=${BATCH_SIZE} \
    --ep=${EPOCHS} \
    --lr=${BASE_LR} \
    --lr=${BASE_LR} \
    --warmup_epochs=${WARMUP_EPOCHS} \
    --warmup_start_lr=${WARMUP_START_LR} \
    --min_lr=${MIN_LR} \
    --weight_decay=${WEIGHT_DECAY} \
    --head_lr_multiplier=${HEAD_LR_MULTIPLIER} \
    --freeze_backbone=${FREEZE_BACKBONE} \
    --grad_clip_norm=${GRAD_CLIP_NORM} \
    --dropout=${DROPOUT_RATE} \
    --num_workers=${NUM_WORKERS} \
    --device="${DEVICE}" \
    --project_name="${PROJECT_NAME}" \
    --exp_name="${EXP_NAME}" \
    --save_path="${SAVE_PATH}" \
    --save_frequency=${SAVE_FREQUENCY} \
    --log_per_iter=${LOG_PER_ITER} \
    --enhanced_augmentation \
    # --crop_padding=28 \
    # --use_mlp_head (如果你的模型有这个选项并且你想在微调时指定)

echo "微调脚本执行完毕。"