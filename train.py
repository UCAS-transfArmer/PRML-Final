import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import datetime

from utils.arg_util import get_args
from dataloader.cifar_dataloader import get_dataloader
from models.resnet import train_resnet

from utils import wandb_utils

from models import (
    LogisticRegression,
    ResNet,
    CNN,
    BoostModel,
    VisionTransformer
)

def get_model(model_name, num_classes, image_dims):

    C, H, W = image_dims
    input_dim_flat = C * H * W # For models that take flattened input

    if model_name == 'logistic':
        model = LogisticRegression(input_dim=input_dim_flat, num_classes=num_classes)
    elif model_name == 'resnet':
        model = ResNet(in_channels=C, num_classes=num_classes, block_config=[9, 9, 9]) 
    elif model_name == 'cnn':
        model = CNN(in_channels=C, num_classes=num_classes, block_config=[9, 9, 9])
    elif model_name == 'boosting':
        model = BoostModel(input_dim=input_dim_flat, num_classes=num_classes, num_estimators=args.num_estimators)
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
        args.dataset, args.bs, args.num_workers, data_root='./data', for_vit=is_vit_model, enhanced_augmentation=True
    )
    print(f"Loaded dataset: {args.dataset} with {num_classes} classes. Image dimensions: {image_dims}")
    
    # Model
    model = get_model(args.model, num_classes, image_dims)
    model.to(device)
    print(f"Loaded model: {args.model}")

    if isinstance(model, BoostModel)==False:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params:,}")

    # Loss function and Optimizer
    criterion = nn.CrossEntropyLoss()
    lr = args.tblr * args.bs / args.bs_base  # Scale learning rate based on batch size
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Starting training for {args.ep} epochs...")
    global_step = 0
    start_time = time.time()

    # Special training for Boosting model
    if isinstance(model, BoostModel):
        print("Using specialized AdaBoost training...")
        start_boosting_time = time.time()

        print("Extracting training data from DataLoader...")
        X_train_list, y_train_list = [], []
        
        for batch_idx, (inputs, labels) in enumerate(trainloader):
            X_train_list.append(inputs)
            y_train_list.append(labels)
        
        X_train = torch.cat(X_train_list, dim=0)
        y_train = torch.cat(y_train_list, dim=0)
        
        print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
        
        model.fit(X_train, y_train, epochs=args.boosting_ep, device=device, test_dataloader=testloader,weak_learner=args.weak_learner_type)
        boosting_duration = time.time() - start_boosting_time
        print(f"Boosting training completed in {datetime.timedelta(seconds=int(boosting_duration))}")

        # Save the Boosting model
        final_model_filename = f'{args.model}_{args.dataset}_final_ep{args.ep}_step{global_step}.pth'
        final_model_path = os.path.join(args.save_path, final_model_filename)
        torch.save({
            'epoch': args.ep,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'args': vars(args)
        }, final_model_path)
        print(f'Saved final Boosting model to {final_model_path}')
        
        return
    elif isinstance(model, (ResNet, CNN)):
        # Special training for ResNet
        global_step = train_resnet(
            args=args,
            epochs=args.ep,
            max_lr=args.max_lr,
            model=model,
            train_loader=trainloader,
            val_loader=testloader,
            device=device,
            weight_decay=args.weight_decay,
            opt_func=torch.optim.AdamW
        )

        # Save the final model
        final_model_filename = f'{args.model}_{args.dataset}_final_ep{args.ep}_step{global_step}.pth'
        final_model_path = os.path.join(args.save_path, final_model_filename)
        torch.save({
            'epoch': args.ep,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'args': args
        }, final_model_path)
        print(f'Saved final model to {final_model_path}')        

    elif isinstance(model, LogisticRegression):

        def evaluate_step(model, dataloader, criterion, device, epoch_num, global_step):

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
            
            epoch_str = f"Epoch {epoch_num} "
            print(f'{epoch_str}Test Set: Avg. Loss: {avg_loss:.4f}, Accuracy: {correct_predictions}/{total_samples} ({accuracy:.2f}%)')
            if args.wandb:
                wandb_utils.log({
                    'test_loss': avg_loss,
                    'test_acc': accuracy
                }, step=global_step)
            
            return avg_loss, accuracy

        for epoch in range(args.ep):
            model.train()  
            running_loss = 0.0
            running_acc = 0.0
            epoch_start_time = time.time()

            for i, (inputs, labels) in enumerate(trainloader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                acc = (outputs.argmax(dim=1) == labels).float().mean().item()
                running_acc += acc
                global_step += 1

                if args.log_per_iter > 0 and global_step % args.log_per_iter == 0:
                    # Calculate average loss over the logging interval
                    avg_loss_interval = running_loss / args.log_per_iter 
                    avg_acc_interval = running_acc / args.log_per_iter

                    print(f'[Epoch {epoch + 1}/{args.ep}, Iter {i + 1}/{len(trainloader)}, Global Step {global_step}] loss: {avg_loss_interval:.4f}')

                    running_loss = 0.0 
                    running_acc = 0.0 
                    
                    if args.wandb:
                        wandb_utils.log({
                            'train_loss': avg_loss_interval,
                            'train_acc': avg_acc_interval,
                        }, step=global_step)
                        

                # Save model checkpoint
                if args.save_per_iter > 0 and global_step % args.save_per_iter == 0:
                    checkpoint_path = os.path.join(args.save_path, f'{args.model}_{args.dataset}_step_{global_step}_{args.exp_name}.pth')
                    torch.save({
                        'global_step': global_step,
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.item(), # Current batch loss
                    }, checkpoint_path)
                    print(f'Saved checkpoint to {checkpoint_path}')
            
            epoch_duration = time.time() - epoch_start_time
            print(f"Epoch {epoch + 1} finished. Duration: {epoch_duration:.2f}s. Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

            evaluate_step(model, testloader, criterion, device, epoch_num=epoch + 1, global_step=global_step)
    else:
        raise ValueError(f"Model '{args.model}' is not supported for training.")

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
   
    if args.wandb:
        wandb_utils.wandb.init(
            project=args.project_name,
            name=args.exp_name,
            config={}
        )
    
    train(args)
