import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import datetime

from utils.arg_util import get_args
from dataloader.cifar_dataloader import get_dataloader

from utils import wandb_utils

from models import (
    LogisticRegression,
    ResNet,
    BoostingModel,
    VisionTransformer
)

def get_model(model_name, num_classes, image_dims):

    C, H, W = image_dims
    input_dim_flat = C * H * W # For models that take flattened input

    if model_name == 'logistic':
        model = LogisticRegression(input_dim=input_dim_flat, num_classes=num_classes)
    elif model_name == 'resnet':
        pass
    elif model_name == 'boosting':
        pass
    elif model_name == 'vit':
        
        model=VisionTransformer(
            image_size=image_dims[1],
            patch_size=16,  #ViT-B/16
            num_classes=num_classes, 
            dim=768, # 隐层维数
            depth=12, # Transformer层数
            heads=12, #注意力头数
            mlp_dim=3072, #MLP维数
            dropout=0.1,
            use_mlp_head=True   
        )
    else:
        raise ValueError(f"Model '{model_name}' not supported.")
    
    return model

def evaluate(model, dataloader, criterion, device, epoch_num=None):

    model.eval()  
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad(): 
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0) 
            _, predicted_labels = torch.max(outputs, 1)
            
            total_samples += labels.size(0)
            correct_predictions += (predicted_labels == labels).sum().item()
            
    avg_loss = total_loss / total_samples
    accuracy = 100.0 * correct_predictions / total_samples
    
    epoch_str = f"Epoch {epoch_num} " if epoch_num is not None else ""
    print(f'{epoch_str}Test Set: Avg. Loss: {avg_loss:.4f}, Accuracy: {correct_predictions}/{total_samples} ({accuracy:.2f}%)')
    return avg_loss, accuracy

def train(args):
    """Main training loop."""

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create save directory if it doesn't exist
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print(f"Created checkpoint directory: {args.save_path}")

    # DataLoaders
    is_vit_model = (args.model == 'vit')
    trainloader, testloader, num_classes, image_dims = get_dataloader(
        args.dataset, args.bs, args.num_workers, data_root='./data', for_vit=is_vit_model
    )
    print(f"Loaded dataset: {args.dataset} with {num_classes} classes. Image dimensions: {image_dims}")
    
    # Model
    model = get_model(args.model, num_classes, image_dims)
    model.to(device)
    print(f"Loaded model: {args.model}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    # Loss function and Optimizer
    criterion = nn.CrossEntropyLoss()
    lr = args.tblr * args.bs / args.bs_base  # Scale learning rate based on batch size
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Starting training for {args.ep} epochs...")
    global_step = 0
    start_time = time.time()

    for epoch in range(args.ep):
        model.train()  
        running_loss = 0.0
        epoch_start_time = time.time()

        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            global_step += 1

            if args.log_per_iter > 0 and global_step % args.log_per_iter == 0:
                # Calculate average loss over the logging interval
                avg_loss_interval = running_loss / args.log_per_iter 
                print(f'[Epoch {epoch + 1}/{args.ep}, Iter {i + 1}/{len(trainloader)}, Global Step {global_step}] loss: {avg_loss_interval:.4f}')
                running_loss = 0.0 # Reset running loss for the next interval

            # Save model checkpoint
            if args.save_per_iter > 0 and global_step % args.save_per_iter == 0:
                checkpoint_path = os.path.join(args.save_path, f'{args.model}_{args.dataset}_step_{global_step}_{args.exp_name}.pth')
                torch.save({
                    'global_step': global_step,
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(), # Current batch loss
                    'args': vars(args) # Save command line arguments
                }, checkpoint_path)
                print(f'Saved checkpoint to {checkpoint_path}')
        
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1} finished. Duration: {epoch_duration:.2f}s. Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

        evaluate(model, testloader, criterion, device, epoch_num=epoch + 1)

    total_training_time = time.time() - start_time
    print(f'Finished Training. Total time: {datetime.timedelta(seconds=int(total_training_time))}')
    
    # Save the final model
    final_model_filename = f'{args.model}_{args.dataset}_final_ep{args.ep}_step{global_step}.pth'
    final_model_path = os.path.join(args.save_path, final_model_filename)
    torch.save({
        'epoch': args.ep,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args)
    }, final_model_path)
    print(f'Saved final model to {final_model_path}')

if __name__ == '__main__':
    args = get_args()
    
    print("Running with the following configuration:")
    for arg_name, value in vars(args).items():
        print(f"  {arg_name}: {value}")
    print("-" * 30)
   
    # Optional: Initialize Weights & Biases logging
    '''
    wandb_utils.initialize(
        args, 
        exp_name=args.exp_name, 
        project_name=args.project_name
    )
    '''
    
    train(args)