# PRML Final Homework

`uv` is recommended to manage the virtual environment.

# Example
To train using logistic regression on the CIFAR-10 dataset, run the following command:

```bash
python train.py --dataset=cifar10 --model=logistic --bs=1024 --ep=50 --tblr=1e-5 --save_path=./ckpts
```

To train using AdaBoost regression on the CIFAR-10 dataset, run the following command:

```bash
python train.py --dataset=cifar10 --model=boosting --bs=1024 --ep=50 --tblr=1e-5 --save_path=./ckpts
```

# Logging Example
`wandb` is highly recommended for logging. If you haven't used it yet, you can sign up a account and refer to the `wandb`  [documentation](https://docs.wandb.ai/quickstart/) for more details.
