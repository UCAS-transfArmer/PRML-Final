# PRML Final Homework

`uv` is recommended to manage the virtual environment.

# Example
To train using logistic regression on the CIFAR-10 dataset, run the following command:

```bash
python train.py --dataset=cifar10 --model=logistic --bs=1024 --ep=50 --tblr=1e-5 --save_path=./ckpts
```

# Logging Example
`wandb` is highly recommended for logging. If you haven't used it yet, you can sign up a account and refer to the `wandb`  [documentation](https://docs.wandb.ai/quickstart/) for more details.

To use wandb in this project, you can use the `wandb_utils.py` as follows:

First, you need to import the `wandb_utils` module:

```python
from utils import wandb_utils
```

Then, you can initialize the wandb logger before training:

```python
wandb_utils.initialize(
    args, 
    exp_name=args.exp_name, 
    project_name=args.project_name
)
```
You are recommended to set `exp_name` for different experiments to help with comparison. For example, if you are training using resnet32 on CIFAR-10 with cosine learning rate scheduler, you can use:
```shell
python train.py --dataset=cifar10 --model=resnet --exp_name=resnet32-cifar10-cosine ...
```

Finally, you can log the metrics during training:

```python
wandb_log_dict = {}
wandb_log_dict['train_loss'] = train_loss
wandb_log_dict['train_acc'] = train_acc
wandb_log_dict['val_loss'] = val_loss
wandb_log_dict['val_acc'] = val_acc
# If you are using a custom metric, you can add it to the log dictionary as well.

wandb_utils.log(wandb_log_dict, step=global_step)
```

For more details, please refer to wandb [documentation](https://docs.wandb.ai/quickstart/). Plus, do not forget to execute `wandb online` before running the training script to enable online synchronization. I prefer to write this in the training script, for example:
```shell
wandb online
python train.py --dataset=cifar10 --model=logistic --bs=1024 --ep=50 --tblr=1e-5 --save_path=./ckpts --exp_name=logistic-cifar10
```
