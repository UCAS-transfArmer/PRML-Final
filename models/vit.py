import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.wandb_utils import initialize, log, log_image

class VisionTransformer(nn.Module):
    """
    标准 ViT-Base/16 配置:
    - 输入图像: 224x224
    - Patch 大小: 16x16
    - 隐层维度: 768
    - MLP 维度: 3072 (4x hidden_dim)
    - 层数: 12
    - 注意力头数: 12
    - 参数量: ~86M
    """
    def __init__(
            self, 
            image_size=224,        # 修改默认值为224
            patch_size=16,         # 修改默认值为16
            num_classes=1000,       
            dim=768,              # 修改为768
            depth=12,             # 修改为12层
            heads=12,             # 修改为12个头
            mlp_dim=3072,         # 修改为3072
            dropout=0.0,
            use_mlp_head=False
        ):
        super(VisionTransformer, self).__init__()
        assert image_size % patch_size == 0, "Image size must be divisible by patch size" 
        
        num_patches = (image_size // patch_size) ** 2  # 对于224/16=196个patch
        patch_dim = 3 * patch_size * patch_size  #3 channels (RGB)
        
        # Patch embedding
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # self.dropout = nn.Dropout(dropout) # Dropout 实例现在只在需要的地方创建

        # Transformer blocks
        self.transformer = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(dim),
                nn.MultiheadAttention(dim, heads, dropout=dropout,batch_first=True),
                #PyTorch的MultiheadAttention默认期望的输入是(序列长度, 批量大小, 特征维度),除非将batch_first设置为True
                nn.LayerNorm(dim),
                nn.Sequential(
                    nn.Linear(dim, mlp_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_dim, dim),
                    nn.Dropout(dropout)
                )
            ]) for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(dim)
        if use_mlp_head:
            #MLP Head
            # 原论文预训练头通常是 Linear -> Activation -> Linear
            # 这里使用一个简单的结构 Linear(dim, hidden_for_head) -> GELU -> Linear(hidden_for_head, num_classes)
            # 为简单起见，且不引入新的超参数，可以使用 dim 作为 hidden_for_head
            self.head=nn.Sequential(
                nn.Linear(dim, dim), # 或者一个特定的 mlp_head_dim
                nn.GELU(),
                # nn.Dropout(dropout), # 从 MLP 头中移除 Dropout，更贴近原论文预训练头
                nn.Linear(dim, num_classes) 
            )
        else:
            #Linear Head(default for pretraining and Fine-tuning)
            self.head=nn.Linear(dim,num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """
        Weights Initialization:
        Positional Embedding 和CLS token使用截断正态分布(std=0.02);
        Linear Layers weights使用截断正态分布,bias初始化为 0;
        Conv Layers权重使用 Xavier 均匀分布，偏置为 0.
        """
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module,nn.LayerNorm):
                if module.elementwise_affine:
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # (B, dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  #展平并交换1,2维度，变成：(B, num_patches, dim)
        
        # Add class token
        b = x.shape[0]
        cls_tokens = self.cls_token.expand(b, -1, -1)  # (B, 1, dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        # x = self.dropout(x) # 移除此处的 Dropout
        
        # Transformer blocks
        for ln1, attn, ln2, mlp in self.transformer:
            # 注意：在 MultiheadAttention 和 MLP 内部已经有 Dropout 了 (如果 dropout > 0)
            x_res = x
            x = ln1(x)
            attn_output, _ = attn(x, x, x) # Multi-head Self Attention
            x = x_res + attn_output

            x_res_mlp = x
            x = ln2(x)
            x = x_res_mlp + mlp(x)  # MLP block
        
        # Classification
        x = self.norm(x)
        x = x[:, 0]  # Take CLS token
        x = self.head(x)
        return x
