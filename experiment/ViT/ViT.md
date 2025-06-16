# PRML Final Project: Vision Transformer (ViT) 

参考论文：[AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE](https://arxiv.org/pdf/2010.11929)

论文解读参考：[论文总结](https://acnn8hqx4j1r

## 实验设计与过程

我们在 CIFAR-10 数据集上进行了以下几组关于 Vision Transformer (ViT) 的实验：

1.  **不同规模 ViT 模型在 CIFAR-10 上的从零开始训练对比**：
    *   **小型 ViT 变体**：6层 Transformer Encoder, 6个注意力头 (heads), 嵌入维度 (dim) 384, MLP内部维度 (mlp_dim) 1536。针对 CIFAR-10 的图像尺寸 (image_size) 32x32，切片大小 (patch_size) 4x4。模型参数量约 11M。
        *   实验结果：在 CIFAR-10 验证集上的准确率达到 82%。
    *   **标准 ViT-Base/16 模型**：12层 Transformer Encoder, 12个注意力头, 嵌入维度 768, MLP内部维度 3072。由于 ViT-Base/16 通常设计用于 image_size 224x224 和 patch_size 16x16，直接在 CIFAR-10 (32x32) 上使用时，我们调整了输入处理或模型配置以适应。模型参数量约 86M。
        *   实验结果：在 CIFAR-10 验证集上的准确率达到 85%。

2.  **ViT-Base/16 在 ImageNet-1K 子集上预训练后于 CIFAR-10 上微调**：
    *   **预训练阶段 (ImageNet-1K 子集)**：
        *   模型：标准 ViT-Base/16 (参数量约 86M)。
        *   数据集：ImageNet-1K (包含约128万张图片，1000类)。考虑到时间和计算资源限制，我们从中随机抽取了约三分之二的图像 (约88万张) 进行预训练。
        *   训练过程：共训练了130个 epoch。
        *   预训练效果：在 ImageNet 子集上，训练准确率 (train_acc) 达到 84%，验证准确率 (val_acc) 达到 56%。尽管在130个 epoch 时模型尚未完全收敛，但已从大规模数据中学到有价值的视觉表征。
    *   **微调阶段 (CIFAR-10)**：
        将在 ImageNet 子集上预训练好的 ViT-Base/16 模型迁移到 CIFAR-10 数据集上进行微调。详细步骤如下：
        1.  **加载预训练模型**：加载在 ImageNet 子集上预训练得到的 ViT 模型权重。
        2.  **调整分类头**：将原始模型的分类头（针对 ImageNet 的1000类）替换为一个新的线性层，其输出维度与 CIFAR-10 数据集的类别数（10个类别）相匹配。
        3.  **全网络参数训练**：在微调过程中，不冻结骨干网络（Transformer 编码器）的参数。模型的所有参数，包括骨干网络和新加的分类头，都将参与梯度的计算和权重的更新。
        4.  **判别式学习率**：采用判别式学习率（Discriminative Fine-tuning）策略。为骨干网络和新添加的分类头设置了不同的学习率，其中分类头的学习率设置为骨干网络学习率的10倍。
        5.  **优化器与学习率调度**：选用 AdamW 优化器，并配合带有预热（Warmup）及后续衰减的学习率调度策略进行训练。
        *   微调效果：在 CIFAR-10 测试集上的准确率达到 95%。

## 实验结果分析与讨论 (基于预期与论文发现)

本部分结合我们的实验结果与原论文的发现，对 ViT 的性能进行分析。

1.  **模型规模与从零训练性能 (CIFAR-10)**：
    *   我们的实验中，小型 ViT (11M) 和标准 ViT-Base/16 (86M) 在 CIFAR-10 上从零训练的准确率分别为 82% 和 85%。
    *   这与原论文的观察基本一致：在中等规模数据集（如 CIFAR-10 或 ImageNet-1K）上从零开始训练时，ViT 的性能可能不如针对小数据集优化的 CNNs，或者需要更多的数据和更强的正则化才能达到SOTA水平。
    *   原论文 Figure 3 显示，ViT 需要更大的数据量才能发挥其相对于 CNN 的优势。Table 2 也提到："When trained on ImageNet-21k, ViT-L/16 achieves 76.5% accuracy on ImageNet, but only 72.7% when trained from scratch on ImageNet-1k."
    *   原文指出："Vision Transformer attains excellent results when pre-trained on large datasets, but performs poorly when trained on insufficient amounts of data."

2.  **ViT (从零训练) vs. ResNet (CIFAR-10)**：
    *   ViT-Base/16 (从零训练) 在 CIFAR-10 上的准确率约为 85%。
    *   相比之下，ResNet-50 等经典 CNN 模型在 CIFAR-10 上的基线准确率通常在 93-95% 左右。
    *   这表明在没有大规模预训练的情况下，传统 CNN 的归纳偏置（如局部性和平移不变性）在中小规模数据集上更具优势。
    *   原论文 Figure 1 清晰地展示了这一现象，并强调其SOTA结果是在大规模预训练的前提下取得的。Section 4.2 提到："We find that Vision Transformers do not generalize well when trained on small datasets without strong regularization."

3.  **预训练对 ViT 性能的影响**：
    *   我们的实验显示，ViT-Base/16 在 ImageNet-1K 子集上预训练后，在 CIFAR-10 上微调的准确率达到了 95%。
    *   相比之下，在 CIFAR-10 上从零训练的 ViT-Base/16 准确率约为 85%。
    *   预训练带来了约 10% 的显著性能提升。
    *   这强有力地支持了原论文的核心观点：大规模预训练是 ViT 成功的关键。Figure 2 展示了预训练规模对性能的影响，原文强调 "large scale pre-training trumps inductive bias." Table 3 也显示了不同预训练数据集规模的效果对比。

## 关键实验洞察

综合我们的实验和原论文的发现，可以总结出以下关键洞察：

1.  **数据量依赖性**：ViT 的性能高度依赖于训练数据的规模。在数据量不足时，其缺乏 CNN 的归纳偏置会导致性能不如传统 CNN。只有在大规模数据集上预训练，ViT 才能充分学习到强大的视觉表征，从而超越 CNN。
2.  **预训练的重要性**：大规模预训练是发挥 ViT 潜力的核心要素。通过在海量数据上学习，ViT 能够克服其结构本身缺乏的图像特有偏置。
3.  **计算效率**：原论文指出，预训练后的 ViT 在达到与顶尖 CNNs 相近性能的同时，其训练所需的计算资源可以显著减少（尤其是在 JFT-300M 这样的大数据集上预训练时）。推理效率也具有竞争力。

> 原文引用："When trained on the largest dataset (JFT-300M), ViT approaches or beats state-of-the-art on multiple image recognition benchmarks... However, when trained on smaller datasets such as ImageNet without strong regularization, these models yield modest accuracies of a few percentage points below ResNet。"

## 实验超参数汇总

### 超参数综合对比表

### 跨实验通用超参数

以下超参数在上述三个实验配置中均明确设置（或对于微调实验隐式继承并保持一致）且采用相同的值：

| 参数名 (Argument)       | 通用值 (Common Value) | 描述 (Description)                                   |
| :---------------------- | :-------------------- | :--------------------------------------------------- |
| `model`                 | `vit`                 | 使用的模型架构                                         |
| `image_size`             | `224`             | `224`                       | `224`                         | 输入图像尺寸 (pixels)                                  |
| `patch_size`             | `16`                          | Patch 尺寸(pixels)                                    |
| `dim`                    | `768`          | Transformer 维度 (embedding dimension)            |
| `depth`                  | `12`           | Transformer 层数 (encoder blocks)                   |
| `heads`                  | `12`           | 多头注意力机制的头数                                  |
| `mlp_dim`                | `3072`        | MLP 内部隐藏层维度                                     |
| `enhanced_augmentation` | `是`                  | 是否使用增强的数据增强策略                                 |
| `num_workers`           | `16`                  | 数据加载使用的工作进程数                                 |
| `use_data_parallel`     | `是`                  | 是否使用 `torch.nn.DataParallel` 进行多GPU训练          |
| `use_amp`               | `是`                  | 是否使用自动混合精度 (Automatic Mixed Precision) 训练 |
| `grad_clip_norm`         | `1.0`       | 梯度裁剪范数                                           |

| 参数名 (Argument)        | 预训练 (ImageNet) | 微调 (CIFAR-10, 基于预训练) | 从零训练 (CIFAR-10, ViT-Base) | 描述 (Description)                                   |
| :---------------------- | :---------------- | :-------------------------- | :---------------------------- | :--------------------------------------------------- |
| `dataset`                | `imagenet`        | `cifar10`                   | `cifar10`                     | 使用的数据集                                           |
| `bs` (batch_size)        | `1600`            | `512`                       | `512`                         | 批处理大小                                             |
| `ep` (epochs)            | `300(实际130)`             | `60(实际20轮收敛)`                        | `200`                         | 训练轮次                                               |
| `lr` (learning_rate)     | `1.6e-3`          | `2e-4`                      | `1e-4`                        | 基础学习率（最大学习率）                                           |
| `warmup_epochs`          | `6`               | `3`                         | `10`                          | 学习率预热轮次数                                         |
| `warmup_start_lr`        | `1e-6`            | `1e-6`                      | `1e-5`                        | 预热起始学习率                                           |
| `min_lr`                 | `1e-5`            | `1e-6`                      | `1e-6`                        | 最小学习率 (学习率衰减下限)                               |
| `dropout`                | `0.0`             | `0.1`                       | `0.1`                         | Dropout 概率                                         |
| `enhanced_augmentation`  | `是`              | `是`                        | `是`                          | 是否使用增强的数据增强策略                                 |
| `weight_decay`           | `0.05`            | `0.05`                      | `0.1`                         | 权重衰减系数                                           |
| `head_lr_multiplier`     | -                 | `10`                        | -                             | 分类头学习率相对于骨干网络学习率的倍数                       |
| `label_smoothing`        | -                 | `0.1`                       | -                             | 标签平滑系数                                             |
| `crop_padding`           | -                 | `28`                        | `4`                           | 随机裁剪时的填充大小 (pixels)                                  |

