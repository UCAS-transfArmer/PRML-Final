uv run train.py \
    --project_name "prml-final" \
    --exp_name "resnet" \
    --dataset "cifar10" \
    --model "resnet" \
    --ep 50 \
    --max_lr 0.01 \
    --weight_decay 1e-4 \
    --save_path "./saves"