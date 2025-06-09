export CUDA_VISIBLE_DEVICES=0
uv run train.py \
    --project_name "prml-final" \
    --exp_name "resnet-ep100-56-layers-data-augmentation" \
    --dataset "cifar10" \
    --model "resnet" \
    --ep 100 \
    --max_lr 0.01 \
    --weight_decay 1e-4 \
    --save_path "./saves"