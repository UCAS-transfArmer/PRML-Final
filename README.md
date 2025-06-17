# PRML Final Homework

[Report Link](https://github.com/UCAS-transfArmer/PRML-Final/blob/main/report.pdf)

# Training Instructions
To train using logistic regression on the CIFAR-10 dataset, run the following command:

```bash
python train.py --dataset=cifar10 --model=logistic --bs=1024 --ep=50 --tblr=1e-5 --save_path=./ckpts
```

To train using AdaBoost regression on the CIFAR-10 dataset, run the following command:

```bash
python train.py --dataset=cifar10 --model=boosting --bs=1024 --ep=50 --tblr=1e-5 --save_path=./ckpts
```

To train ResNet on the CIFAR-10 dataset, run the following script:

```bash
./scripts/train_resnet.sh
```

To run the ViT-model codes, you can run the following command to switch to the 'ViT-Specific' branch:
```bash
git checkout ViT-Specific
```

Suppose you're on the 'ViT-Specific' branch, run the following commands to train ViT model on cifar from scratch:
```bash
chmod +x ./scripts/train_vit_cifar.sh
./scripts/train_vit_cifar.sh
```
To pre-train ViT model on ImageNet-1K, run the following command to download ImageNet-1K dataset from Huggingface:
```bash
python download_hf_imagenet.py
```
Then run the following command for pre-training:
```bash
chmod +x ./scripts/pretrain_vit_imagenet.sh
./scripts/pretrain_vit_imagenet.sh
```
Make sure your path of checkpoints is correct, then run the following command for finetuning:
```bash
chmod +x ./scripts/finetune_vit_cifar.sh
./scripts/finetune_vit_cifar.sh
```
