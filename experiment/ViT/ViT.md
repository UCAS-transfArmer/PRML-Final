Vision Transformer (ViT) Summary and Experimental Plan

贴上原论文链接：[《AN IMAGE IS WORTH 16X16 WORDS:  TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE》](：https://arxiv.org/pdf/2010.11929)
1. Method
1.1 输入
1.1.1 图像输入

(\boldsymbol{x} \in \mathcal{R}^{H \times W \times C}): ((H, W)) 是图片横纵方向的像素点个数，(C) 是 channel 数
把它 reshape 成：
(\boldsymbol{x_p} \in \mathcal{R}^{N \times (P^2 \times C)}): ((P, P)) 是每个 patch 的大小，其中 (P) 是 patch 的边长，以像素为单位
(N = \frac{HW}{P^2}): (N) 是 ViT 等效的输入序列长度



1.1.2 展平 patch

每个 patch 展平并通过线性层映射为 (D) 维向量（(D) 是模型的隐藏维度）
核心功能：[CLS] token 是一个额外的可学习嵌入向量，添加到由图像 patch 转换而来的 token 序列中。它在 Transformer 的自注意力机制中与所有 patch token 交互，充当一个“全局代表”，用于捕获整个图像的全局特征。
工作原理：在 ViT 中，输入图像被分割成 (N) 个 patch，每个 patch 展平并映射为一个固定维度的向量（token）。[CLS] token 作为一个额外的 token（通常放在序列的开头），通过多层 Transformer 的自注意力机制，与所有 patch token 交互，逐步聚合整个图像的上下文信息。

Q: 为什么一定要先分 patch，再从 patch 转 token 呢？A:

减少模型计算量。在 Transformer 中，假设输入的序列长度为 (N)，那么经过 attention 时，计算复杂度就为 (O(N^2))，因为注意力机制下，每个 token 都要和包括自己在内的所有 token 做一次 attention score 计算。在 ViT 中复杂度是 (O(N^2))，当 patch 尺寸 (P) 越小时，(N) 越大，此时模型的计算量也就越大。因此，我们需要找到一个合适的 (P) 值，来减少计算压力。
图像数据带有较多的冗余信息。和语言数据中蕴含的丰富语义不同，像素本身含有大量的冗余信息。比如，相邻的两个像素格子间的取值往往是相似的。因此我们并不需要特别精准的计算粒度（比如把 (P) 设为 1）。(这个特性也是之后 MAE，MoCo 之类的像素级预测模型能够成功的原因之一。）

1.1.3 加 [CLS] 作为 patch 的“全局代表”

借鉴了 BERT 模型的思路，在序列前面加了一个可学习的 [class] token（也是 (D) 维），然后给所有的 token 加上位置编码；
在 ViT 的 pretrain 和 finetune 两个阶段，都会在 Transformer 的最后一层输出中，使用 [CLS] token 的输出向量（记为 (z_L^0)）连接一个分类头（classification head）：
pretain 阶段的分类头是一个带隐藏层的 MLP；提供非线性能力，帮助模型捕捉复杂的模式
fine-tune 阶段是一个单一的线性层：参数少，成本低，防止过拟合


即：(TransformerEncoder(z^0_L) = \boldsymbol{y})

1.1.4 Positional Embedding

作者发现用 1D 位置编码的效果要好于 2D，所以直接用 1D

1.1.5 Transformer Encoder

通过每一层的公式：

[图片]  

Inductive Bias (归纳偏见)：
ViT 的弱归纳偏见：Transformer 依赖自注意力机制，几乎没有强加关于数据结构的先验假设。图像方面：ViT 将图像分割成 patch 并按序列处理，自注意力机制允许模型学习任意 patch 之间的关系，不假设局部性（locality）或平移不变性（translation invariance），较为灵活；缺点就是数据需求高：由于归纳偏见较弱，ViT 需要大量数据来学习图像的结构信息（如局部相关性），否则性能可能不如 CNN。
CNNs 的强归纳偏见：
局部性：卷积操作假设图像中相邻像素更相关，因此只关注局部区域。
平移不变性：卷积核在图像上滑动，参数共享使模型对物体的位置不敏感。


优势：  
数据效率高：这些假设与图像的自然属性契合，因此 CNN 在小数据集上也能表现良好。
计算效率：卷积操作比自注意力更轻量，适合较小的计算资源。


缺点：  
全局依赖不足：CNN 难以捕捉远距离的像素关系，除非通过深层堆叠或池化操作。






Hybrid Architecture (混合架构)：作为原始图像补丁的替代方案，输入序列可以由 CNN 的特征映射形成（LeCun et al, 1989）。在该混合模型中，将 patch embedding 投影 (E)（Eq. 1）应用于从 CNN feature map 中提取的 patch。  
优点：融合 CNN 的局部特征提取能力和 Transformer 的全局建模能力。



2. Experiments

作者对比了 ViT 和 CNN（卷积神经网络）在不同数据集上的表现：
在比较小的数据集上训练，ViT 不优于 ResNet（CNN 模型）；而在大数据集上，ViT 效果优于 ResNet；
在比较大的数据集上做预训练后，然后在小数据集上做 fine-tune，效果也可以超过 ResNet。



PRML-Final -- ViT
做两部分实验：
1. 用 ViT 在 CIFAR-10（train+val 共 6 万张，10 类）上训练和验证：

对比两组参数：
一个小的 ViT 变体，6 层，heads=6, dim=384, image_size=32, patch_size=4, mlp_dim=1536，参数量约 11M进度：跑了一次 val_acc 只有 82%，有空再跑一次看看能不能高点
标准的 ViT-base/16，12 层，heads=12, dim=768, image_size=224, patch_size=16, mlp_dim=3072，参数量约 86M，进度：还在调参数



2. 用标准 ViT-base/16 在 ImageNet-1K (train+val 约 128 万张图片，1000 类)，再在 CIFAR-10 上微调

进度：pretrain 已经写好, finetune 在调试

期望的实验结果:
1. 小型 ViT vs 标准 ViT-Base/16（CIFAR-10 从头训练）

小型 ViT（11M 参数）：准确率约 82-85%
标准 ViT-Base/16（86M 参数）
两者表现相近，但都显著低于 ResNet 基线
符合原论文中提到的观点：
Figure 3 显示，在中等规模数据集上，ViT 需要更大的数据量才能发挥优势
Table 2 中提到："When trained on ImageNet-21k, ViT-L/16 achieves 76.5% accuracy on ImageNet, but only 72.7% when trained from scratch on ImageNet-1k"
原文指出："Vision Transformer attains excellent results when pre-trained on large datasets, but performs poorly when trained on insufficient amounts of data"



2. 未经预训练的 ViT vs ResNet 对比

ViT-Base/16（从头训练）：~85-88%
ResNet-50：~93-95%（CIFAR-10 标准基线）
ResNet 明显优于 ViT
原论文：
原论文 Figure 1 清楚地展示了这一现象："Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train"，但这是在大规模预训练的前提下
原文 Section 4.2 提到："We find that Vision Transformers do not generalize well when trained on small datasets without strong regularization"
Table 1 显示 ResNet 在中小规模数据集上的优势



3. 预训练 ViT vs 从头训练 ViT

在 ImageNet-1K 上预训练后在 CIFAR-10 微调：~95%（？最好能达到）
CIFAR-10 从头训练：~85-88%
预训练带来显著提升（~10%）
原论文：
Figure 2 展示了预训练规模对性能的影响
原文强调："large scale pre-training trumps inductive bias"
Table 3 显示了不同预训练数据集规模的效果对比



4. 关键实验洞察

数据量依赖性：ViT 需要大规模数据才能超越 CNN 的归纳偏置优势
预训练重要性：大规模预训练是 ViT 成功的关键
计算效率：预训练后的 ViT 在推理时比同等性能的 CNN 更高效


原文："When trained on the largest dataset (JFT-300M), ViT approaches or beats state-of-the-art on multiple image recognition benchmarks... However, when trained on smaller datasets such as ImageNet without strong regularization, these models yield modest accuracies of a few percentage points below ResNet."

