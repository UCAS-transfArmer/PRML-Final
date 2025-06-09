import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import wandb_utils

#########################################
#           The Residual block          #
#########################################

class ResidualBlock(nn.Module):
    """More standard Residual Block."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = conv_block(in_channels, out_channels, stride=stride)
        self.conv2 = conv_block(out_channels, out_channels, stride=1)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.gelu(out)
        return out


#########################################
#           The Basic CNN Block         #
#########################################

class BasicBlock(nn.Module):
    """A basic block with two convolutional layers, without residual connection."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # The first conv_block handles the stride for downsampling
        self.conv1 = conv_block(in_channels, out_channels, stride=stride)
        # The second conv_block always has stride=1
        self.conv2 = conv_block(out_channels, out_channels, stride=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out


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
    Configurable ResNet architecture.
    For ResNet-20 (19 conv + 1 FC): block_config=[3, 3, 3]
    For ResNet-32 (31 conv + 1 FC): block_config=[5, 5, 5]
    For ResNet-56 (55 conv + 1 FC): block_config=[9, 9, 9]
    Each ResidualBlock contains 2 convolutional layers.
    """

    def __init__(self, in_channels, num_classes, block_config=[3,3,3], channels_config=[64, 128, 256]):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        assert len(block_config) == 3, "block_config should have 3 elements for 3 stages."
        assert len(channels_config) == 3, "channels_config should have 3 elements for 3 stages."

        # Initial convolutional layer
        # The first element of channels_config defines the output channels of the initial conv and the first stage
        initial_conv_out_channels = channels_config[0]

        self.conv1 = conv_block(in_channels, initial_conv_out_channels, pool=False, stride=1)
        current_channels = initial_conv_out_channels

        # Stage 1
        self.stage1 = self._make_layer(ResidualBlock, current_channels, channels_config[0], block_config[0], stride=1)

        # Stage 2
        self.stage2 = self._make_layer(ResidualBlock, channels_config[0], channels_config[1], block_config[1], stride=2)
        current_channels = channels_config[1]

        # Stage 3
        self.stage3 = self._make_layer(ResidualBlock, current_channels, channels_config[2], block_config[2], stride=2)
        current_channels = channels_config[2]

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(current_channels, num_classes)
    
    def _make_layer(self, block_class, in_channels_stage, out_channels_stage, num_blocks, stride):
        """
        Creates a stage of residual blocks.
        - block_class: The class of the residual block to use (e.g., ResidualBlock).
        - in_channels_stage: Input channels to the first block of this stage.
        - out_channels_stage: Output channels from all blocks in this stage.
        - num_blocks: Number of residual blocks in this stage.
        - stride: Stride for the first block of this stage (for downsampling).
        """
        layers = []
        # First block: handles potential downsampling (stride) and change in channel dimensions
        layers.append(block_class(in_channels_stage, out_channels_stage, stride=stride))

        # Subsequent blocks: maintain channel dimensions (out_channels_stage -> out_channels_stage) and use stride=1
        for _ in range(1, num_blocks):
            layers.append(block_class(out_channels_stage, out_channels_stage, stride=1))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


#################################################################
#           The CNN class (witout residual connections)         #
#################################################################

class CNN(ImageClassificationBase):
    """
    Configurable CNN architecture, similar to ResNet structure but without residual connections.
    Uses BasicBlock instead of ResidualBlock.
    """

    def __init__(self, in_channels, num_classes, block_config=[3,3,3], channels_config=[64, 128, 256]):
        super().__init__()

        assert len(block_config) == 3, "block_config should have 3 elements for 3 stages."
        assert len(channels_config) == 3, "channels_config should have 3 elements for 3 stages."

        # Initial convolutional layer
        initial_conv_out_channels = channels_config[0]
        self.conv1 = conv_block(in_channels, initial_conv_out_channels, pool=False, stride=1)
        
        current_channels = initial_conv_out_channels

        # Stage 1
        self.stage1 = self._make_layer(BasicBlock, current_channels, channels_config[0], block_config[0], stride=1)

        # Stage 2
        self.stage2 = self._make_layer(BasicBlock, channels_config[0], channels_config[1], block_config[1], stride=2)
        current_channels = channels_config[1]

        # Stage 3
        self.stage3 = self._make_layer(BasicBlock, current_channels, channels_config[2], block_config[2], stride=2)
        current_channels = channels_config[2]

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(current_channels, num_classes)

    def _make_layer(self, block_class, in_channels_stage, out_channels_stage, num_blocks, stride):
        """
        Creates a stage of basic blocks.
        - block_class: The class of the basic block to use (e.g., BasicBlock).
        - in_channels_stage: Input channels to the first block of this stage.
        - out_channels_stage: Output channels from all blocks in this stage.
        - num_blocks: Number of basic blocks in this stage.
        - stride: Stride for the first block of this stage (for downsampling).
        """
        layers = []
        # First block: handles potential downsampling (stride) and change in channel dimensions.
        layers.append(block_class(in_channels_stage, out_channels_stage, stride=stride))
        
        # Subsequent blocks: maintain channel dimensions (out_channels_stage -> out_channels_stage) and use stride=1.
        for _ in range(1, num_blocks):
            layers.append(block_class(out_channels_stage, out_channels_stage, stride=1))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
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

def conv_block(in_channels, out_channels, pool=False, stride=1):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.GELU()
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

def train_resnet(args, epochs, max_lr, model, train_loader, val_loader, device, weight_decay=0, grad_clip=None, opt_func=torch.optim.AdamW):
    # Initialize wandb
    wandb_utils.initialize(
        args,
        entity="ericguoxy-ucas",
        exp_name=args.exp_name,
        project_name=args.project_name
    )

    torch.cuda.empty_cache()

    # Wrap train_loader and val_loader with DeviceDataLoader
    train_loader = DeviceDataLoader(train_loader, device)
    val_loader = DeviceDataLoader(val_loader, device)

    # Set up a custom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, div_factor=25, epochs=epochs, steps_per_epoch=len(train_loader))

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
            sched.step()

            # Record & update learning rate
            lrs.append(get_lr(optimizer))
        
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result, global_steps)
    return global_steps