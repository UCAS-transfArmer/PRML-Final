wandb online
python train.py \
    --dataset=cifar10 \
    --model=logistic \
    --bs=1024 \
    --ep=100 \
    --tblr=1e-5 \
    --save_path=./ckpts \
    --exp_name=logistic \
    --save_per_iter=1000 \
    --wandb=1
