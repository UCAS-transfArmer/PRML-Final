import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from utils.wandb_utils import initialize, log, log_image

class VisionTransformer(nn.Module):

    """
    ViT-Base/16 
    """

    def __init__(self, image_size=224, patch_size=16, num_classes=10, dim=768, depth=12, heads=12, mlp_dim=3072, dropout=0.1,use_mlp_head=False):
        super(VisionTransformer, self).__init__()
        assert image_size % patch_size == 0, "Image size must be divisible by patch size" 
        
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size  #3 channels (RGB)
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.transformer = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(dim),
                nn.MultiheadAttention(dim, heads, dropout=dropout),
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
            #MLP head(optional for pretraining or scratch)
            self.head=nn.Sequential(
                nn.Linear(dim,dim//2),#768->384
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim//2,num_classes) #384->num_classes
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
        x = self.dropout(x)
        
        # Transformer blocks
        for ln1, attn, ln2, mlp in self.transformer:
            x = ln1(x)
            x = x + attn(x, x, x)[0]  # Multi-head Self Attention
            x = ln2(x)
            x = x + mlp(x)  # MLP block
        
        # Classification
        x = self.norm(x)
        x = x[:, 0]  # Take CLS token
        x = self.head(x)
        return x

def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, device, args, run_name, pretrain=False):
    # Initialize wandb
    initialize(args, entity="your_wandb_entity", exp_name=run_name, project_name="vit-cifar10")
    
    model.to(device)
    global_step = 0
    for epoch in range(args.ep):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            global_step += 1
            
            if args.log_per_iter > 0 and global_step % args.log_per_iter == 0:
                avg_loss = running_loss / args.log_per_iter
                accuracy = 100. * correct / total
                stats = {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "train_loss": avg_loss,
                    "train_accuracy": accuracy
                }
                log(stats, step=global_step)
                log_image("train_samples", inputs[:16], step=global_step)
                running_loss = 0.0
                correct = 0
                total = 0
        
        # Evaluate after each epoch
        model.eval()
        test_loss, test_acc = 0.0, 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        test_loss /= len(test_loader.dataset)
        test_acc = 100. * correct / total
        stats = {
            "epoch": epoch + 1,
            "test_loss": test_loss,
            "test_accuracy": test_acc
        }
        log(stats, step=global_step)
        
        # Save checkpoint
        if args.save_per_iter > 0 and global_step % args.save_per_iter == 0:
            checkpoint_path = os.path.join(args.save_path, f'{args.model}_{args.dataset}_step_{global_step}.pth')
            torch.save({
                'global_step': global_step,
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                'args': vars(args)
            }, checkpoint_path)
    
    # Save final model
    final_path = os.path.join(args.save_path, f'{args.model}_{args.dataset}_final_ep{args.ep}_step{global_step}.pth')
    torch.save({
        'epoch': args.ep,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args)
    }, final_path)
    
    # Log final test accuracy to wandb
    log({"final_test_accuracy": test_acc}, step=global_step)
    return test_acc

if __name__ == "__main__":
    from utils.arg_util import get_args
    from dataloader.cifar_dataloader import get_dataloader
    
    args = get_args()
    device = torch.device(args.device)
    
    # Scenario 1: Train ViT-Base on CIFAR-10 without pretraining
    train_loader, test_loader, num_classes, image_dims = get_dataloader(
        args.dataset, args.bs, args.num_workers, data_root='./data', for_vit=True
    )
    model = VisionTransformer(
        image_size=image_dims[1],  # Use H from image_dims (224 for ViT)
        patch_size=16,
        num_classes=num_classes,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.tblr * args.bs / args.bs_base)
    print("Scenario 1: Training ViT-Base on CIFAR-10 without pretraining")
    test_acc1 = train_and_evaluate(
        model, train_loader, test_loader, criterion, optimizer, device, args,
        run_name="vit-base-cifar10-no-pretrain"
    )
    print(f"Scenario 1 Final Test Accuracy: {test_acc1:.2f}%")
    
    # Scenario 2: Train ViT-Base on CIFAR-10 with data augmentation (already in dataloader)
    train_loader, test_loader, num_classes, image_dims = get_dataloader(
        args.dataset, args.bs, args.num_workers, data_root='./data', for_vit=True
    )
    model = VisionTransformer(
        image_size=image_dims[1],
        patch_size=16,
        num_classes=num_classes,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.tblr * args.bs / args.bs_base)
    print("\nScenario 2: Training ViT-Base on CIFAR-10 with data augmentation")
    test_acc2 = train_and_evaluate(
        model, train_loader, test_loader, criterion, optimizer, device, args,
        run_name="vit-base-cifar10-augmented"
    )
    print(f"Scenario 2 Final Test Accuracy: {test_acc2:.2f}%")
    
    # Scenario 3: Pretrain ViT-Base on ImageNet-10k, fine-tune on CIFAR-10
    # Note: ImageNet-10k is not standard; placeholder used. Replace with actual dataset.
    model = VisionTransformer(
        image_size=224,
        patch_size=16,
        num_classes=10000,  # Assuming ImageNet-10k has 10,000 classes
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.tblr * args.bs / args.bs_base)
    # Placeholder for ImageNet-10k (replace with actual data loading)
    print("\nScenario 3: Pretraining ViT-Base on ImageNet-10k (placeholder)")
    # imagenet_loader, _, _, _ = get_dataloader(
    #     'imagenet10k', args.bs, args.num_workers, data_root='./data/imagenet', for_vit=True
    # )
    # test_acc_pretrain = train_and_evaluate(
    #     model, imagenet_loader, imagenet_loader, criterion, optimizer, device, args,
    #     run_name="vit-base-imagenet10k-pretrain", pretrain=True
    # )
    
    # Fine-tune on CIFAR-10
    model = VisionTransformer(
        image_size=image_dims[1],
        patch_size=16,
        num_classes=num_classes,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.tblr * args.bs / args.bs_base / 10)  # Lower LR for fine-tuning
    print("Fine-tuning ViT-Base on CIFAR-10")
    test_acc3 = train_and_evaluate(
        model, train_loader, test_loader, criterion, optimizer, device, args,
        run_name="vit-base-imagenet10k-finetune-cifar10"
    )
    print(f"Scenario 3 Final Test Accuracy: {test_acc3:.2f}%")