# train_logistic.sh

# 启用 WandB Online 模式
wandb online

# 训练命令，新增 --entity 参数
python train.py \
    --model=logistic \
    --bs=1024 \
    --ep=50 \
    --tblr=1e-5 \
    --save_path=./ckpts \