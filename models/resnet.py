import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import os
from ..utils import wandb_utils
from ..dataloader.cifar_dataloader import get_cifar10_dataloader

#########################################
#           The Residual block          #
#########################################

class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.gelu1 = nn.GELU()
        self.conv2 = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.gelu2 = nn.GELU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.gelu1(out)
        out = self.conv2(out)
        return self.gelu2(out) + x


#############################################
#           The DataLoader class            #
#############################################

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device."""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    
    def __iter__(self):
        """Yield a batch of data after moving it to device."""
        for b in self.dl:
            yield to_device(b, self.device)
    
    def __len__(self):
        """Number of batches."""
        return len(self.dl)


#################################################
#           The base class for ResNet           #
#################################################

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images) # Generate predictions
        loss = F.cross_entropy(out, labels)
        self.global_steps += 1
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()

        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result, global_steps):
        wandb_utils.log({
            'last_lr': result['lrs'][-1],
            'train_loss': result['train_loss'],
            'val_loss': result['val_loss'],
            'val_acc': result['val_acc']
        }, step=global_steps)
        print(f"Epoch [{epoch}], last_lr: {result['lrs'][-1]:.5f}, train_loss: {result['train_loss']:.4f}, val_loss: {result['val_loss']:.4f}, val_acc: {result['val_acc']:.4f}")


#########################################
#           The ResNet class            #
#########################################

class ResNet(ImageClassificationBase):
    """
    ResNet-9
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


#########################################
#           Helper functions            #
#########################################

def to_device(data, device):
    """Move tensor(s) to chosen deivce."""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def conv_block(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.GELU(inplace=True)
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_resnet(args, epochs, max_lr, model, train_loader, val_loader, weight_decay=0, grad_clip=None, opt_func=torch.optim.AdamW):
    # Initialize wandb
    wandb_utils.initialize(
        args,
        exp_name=args.exp_name,
        project_name=args.project_name
    )

    torch.cuda.empty_cache()

    # Set up a custom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))

    global_steps = 0

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            global_steps += 1

            optimizer.zero_grad()

            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()

            # Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()

            # Record & update learning rate
            lrs.append(get_lr(optimizer))
        
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result, global_steps)

    # Save the final model
    final_model_filename = f'{args.model}_{args.dataset}_final_ep{args.ep}_step{global_steps}.pth'
    final_model_path = os.path.join(args.save_path, final_model_filename)
    torch.save({
        'epoch': args.ep,
        'global_step': global_steps,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args)
    }, final_model_path)
    print(f'Saved final model to {final_model_path}')