# PRML-Final -- ViT：


贴上原论文链接：[《AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE》](https://arxiv.org/pdf/2010.11929)

这里放一个[论文总结](https://acnn8hqx4j1r.feishu.cn/docx/JFdIdMfCJo26OrxhkoPcOOIOnah)
## 做两部分实验：

1.  用ViT在CIFAR-10（train+val共6万张，10类）上训练和验证：

    对比两组参数：
    
    -   一个小的ViT变体 ，6层，heads=6, dim=384, image_size=32, patch_size=4, mlp_dim=1536，参数量约11M

        效果：val_acc=82%

    -   标准的ViT-base/16 ，12层，heads=12, dim=768, image_size=224, patch_size=16, mlp_dim=3072，参数量约86M，

        效果：val_acc 85%

3.  用标准ViT-base/16在ImageNet-1K(train+val约128万张图片，1000类)，再在CIFAR-10上微调。

 实际实验时，由于时间和设备条件有限，我们在ImageNet-1K的128万张图片中随机挑选出三分之二（约88万张），对ViT-base/16模型（86M）进行预训练了130个epoch。
 
- 预训练效果：train_acc=84%, val_acc=56%（130个epoch时还远未收敛，但模型已经在ImageNet上学到很多有用的特征）

- 微调后在CIFAR-10上的测试效果：95%

## 期望的实验结果:

1.  小型ViT vs 标准ViT-Base/16（CIFAR-10从头训练）

    -   小型ViT（11M参数）：准确率约 82-85%
  
    -   标准ViT-Base/16（86M参数）
      
    -   两者表现相近，但都显著低于ResNet基线
    
    符合原论文中提到的观点：

    -   Figure 3显示，在中等规模数据集上，ViT需要更大的数据量才能发挥优势

    -   Table 2中提到：

    -   "When trained on ImageNet-21k, ViT-L/16 achieves 76.5% accuracy on ImageNet, but only 72.7% when trained from scratch on ImageNet-1k"

    -   原文指出：

    -   "Vision Transformer attains excellent results when pre-trained on large datasets, but performs poorly when trained on insufficient amounts of data"

3.  未经预训练的ViT vs ResNet对比

    -   ViT-Base/16（从头训练）：~85-88%

    -   ResNet-50：~93-95%（CIFAR-10标准基线）

    -   ResNet明显优于ViT
    
    原论文：

    -   原论文Figure 1清楚地展示了这一现象：

    -   "Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train"，但这是在大规模预训练的前提下

    -   原文Section 4.2提到：

    -   "We find that Vision Transformers do not generalize well when trained on small datasets without strong regularization"

    -   Table 1显示ResNet在中小规模数据集上的优势

5.  预训练ViT vs 从头训练ViT

    -   在ImageNet-1K上预训练后在CIFAR-10微调：~95%（？最好能达到）

    -   CIFAR-10从头训练：~85-88%

    -   预训练带来显著提升（~10%）
    
    原论文：

    -   Figure 2展示了预训练规模对性能的影响

    -   原文强调：

    -   "large scale pre-training trumps inductive bias"

    -   Table 3显示了不同预训练数据集规模的效果对比

7.  关键实验洞察

    1.  数据量依赖性：ViT需要大规模数据才能超越CNN的归纳偏置优势

    2.  预训练重要性：大规模预训练是ViT成功的关键

    3.  计算效率：预训练后的ViT在推理时比同等性能的CNN更高效
    
    原文：
    
    "When trained on the largest dataset (JFT-300M), ViT approaches or beats state-of-the-art on multiple image recognition benchmarks... However, when trained on smaller datasets such as ImageNet without strong regularization, these models yield modest accuracies of a few percentage points below ResNet."
