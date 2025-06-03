# PRML Final Homework

`uv` is recommended to manage the virtual environment.

`wandb` is recommended for logging.

# Example
To train using logistic regression on the CIFAR-10 dataset, run the following command:

```bash
python train.py --dataset=cifar10 --model=logistic --bs=1024 --ep=50 --tblr=1e-5 --save_path=./ckpts
```