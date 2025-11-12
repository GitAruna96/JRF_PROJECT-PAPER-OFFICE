import os
import sys
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
import torch.fft as fft
from sac_data_load import get_loader
from timm import create_model
import time
from torchvision import transforms
import warnings
warnings.filterwarnings('ignore')

# Enhanced Spectral Perturbation with gradual frequency weighting
def spectral_perturbation(x, noise_scale=0.03, high_freq_ratio=0.8):
    """
    Enhanced spectral perturbation with gradual frequency weighting
    """
    freq = fft.fft2(x)
    freq_shift = fft.fftshift(freq)
    h, w = x.shape[-2:]
    cy, cx = h // 2, w // 2
    
    # Create gradual frequency mask using distance from center
    y, x = torch.meshgrid(torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing='ij')
    dist_from_center = torch.sqrt((y - cy).float()**2 + (x - cx).float()**2)
    max_dist = torch.sqrt(torch.tensor(cy**2 + cx**2, device=x.device).float())
    
    # Gradual weighting - higher frequencies get more noise
    weight = torch.clamp((dist_from_center - (max_dist * 0.3)) / (max_dist * 0.5), 0, 1)
    mask = weight.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    
    noise = noise_scale * torch.randn_like(freq_shift) * mask
    freq_perturbed = freq_shift + noise
    freq_back = fft.ifftshift(freq_perturbed)
    x_perturbed = fft.ifft2(freq_back).real
    x_perturbed = torch.clamp(x_perturbed, 0, 1)
    
    return x_perturbed

def train_epoch(model, src_train_loader, trg_train_loader, optimizer, 
                criterion_cls, conf_threshold, lambda_sac, device, epoch):
    """Enhanced training for single-source domain adaptation"""
    
    model.train()
    total_cls_loss = 0.0
    total_sac_loss = 0.0
    num_batches = 0

    # Dynamic lambda_sac based on epoch (gradual warmup)
    current_lambda = lambda_sac * min(epoch / 10, 1.0)

    for src_batch, trg_batch in zip(src_train_loader, trg_train_loader):
        src_imgs, src_labels = src_batch[0].to(device), src_batch[1].to(device)
        trg_imgs = trg_batch[0].to(device)

        # Source classification (supervised)
        src_outputs = model(src_imgs)
        cls_loss = criterion_cls(src_outputs, src_labels)

        # Target: Enhanced SAC with multiple perturbations
        trg_outputs = model(trg_imgs)
        trg_probs = torch.softmax(trg_outputs, dim=1)
        conf_scores, pseudo_labels = torch.max(trg_probs, dim=1)
        anchor_mask = conf_scores > conf_threshold
        
        sac_loss = torch.tensor(0.0, device=device)
        if anchor_mask.sum() > 0:
            anchor_imgs = trg_imgs[anchor_mask]
            anchor_outputs = trg_outputs[anchor_mask]
            
            # Apply multiple perturbations for robustness
            perturbations = []
            for i in range(2):  # Two different perturbations
                current_noise_scale = 0.02 + 0.01 * random.random()
                perturbed = spectral_perturbation(anchor_imgs, noise_scale=current_noise_scale)
                perturbations.append(perturbed)
            
            total_consistency_loss = 0.0
            for perturbed_anchor in perturbations:
                perturbed_outputs = model(perturbed_anchor)
                
                # Use KL divergence for better probability distribution matching
                consistency_loss = nn.KLDivLoss(reduction='batchmean')(
                    F.log_softmax(perturbed_outputs, dim=1),
                    F.softmax(anchor_outputs.detach(), dim=1)
                )
                total_consistency_loss += consistency_loss
            
            sac_loss = total_consistency_loss / len(perturbations)

        # Combined loss with dynamic weighting
        total_loss = cls_loss + current_lambda * sac_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()

        total_cls_loss += cls_loss.item()
        total_sac_loss += sac_loss.item() if anchor_mask.sum() > 0 else 0.0
        num_batches += 1

    avg_cls_loss = total_cls_loss / num_batches if num_batches > 0 else 0.0
    avg_sac_loss = total_sac_loss / num_batches if num_batches > 0 else 0.0

    return avg_cls_loss, avg_sac_loss

def evaluate(model, eval_loader, device, criterion=None):
    """Evaluate model performance"""
    
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0 if criterion else None
    
    with torch.no_grad():
        for images, labels in eval_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            if criterion:
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total if total > 0 else 0.0
    avg_loss = total_loss / len(eval_loader) if criterion and len(eval_loader) > 0 else None
    
    return accuracy, avg_loss

def save_history_to_csv(history, model_name, output_dir):
    """Save training history to CSV"""
    
    epochs = list(range(1, len(history['train_cls_loss']) + 1))
    df = pd.DataFrame({
        'Epoch': epochs,
        'Train_CLS_Loss': history['train_cls_loss'],
        'Train_SAC_Loss': history['train_sac_loss'],
        'Test_Accuracy': history['test_acc'],
        'Test_Loss': history['test_loss'],
        'Conf_Threshold': history['conf_threshold'],
        'Learning_Rate': history['learning_rate'],
        'Epoch_Time_s': history['epoch_time'],  
        'Cumulative_Time_s': history['cumulative_time']  
    })
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f'{model_name}_improved_results_1__w_a_{timestamp}.csv')
    
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")
    return csv_path

def save_checkpoint(model, optimizer, scheduler, history, epoch, test_acc, conf_threshold, output_dir, model_name):
    """Save complete training state"""
    
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'test_accuracy': test_acc,
        'conf_threshold': conf_threshold,
        'history': history,
        'random_state': {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
        }
    }
    
    if torch.cuda.is_available():
        checkpoint['random_state']['cuda'] = torch.cuda.get_rng_state_all()
    
    filename = os.path.join(output_dir, f'checkpoint_1_epoch_w_a_{epoch+1}_{test_acc:.2f}.pth')
    torch.save(checkpoint, filename)
    print(f"✓ Checkpoint saved: {filename}")
    return filename

def get_enhanced_transforms(train=True):
    """Enhanced data transformations with strong augmentations"""
    
    if train:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform

class DomainAdaptationModel(nn.Module):
    """Enhanced model with feature extraction capability"""
    
    def __init__(self, backbone_name, num_classes, pretrained=True):
        super().__init__()
        self.backbone = create_model(backbone_name, pretrained=pretrained, num_classes=num_classes)
        
        # Get feature dimension
        if hasattr(self.backbone, 'num_features'):
            self.feature_dim = self.backbone.num_features
        else:
            # For Swin Transformer, we need to check the classifier
            if hasattr(self.backbone, 'head'):
                self.feature_dim = self.backbone.head.in_features
            else:
                self.feature_dim = 768  # Default for Swin Tiny
        
        # Replace classifier
        self.backbone.reset_classifier(num_classes)
        
    def forward(self, x, return_features=False):
        features = self.backbone.forward_features(x)
        
        if return_features:
            return features
            
        if hasattr(self.backbone, 'head'):
            x = self.backbone.head(features)
        else:
            # Global average pooling if no head exists
            x = features.mean(dim=1)
            if hasattr(self.backbone, 'fc'):
                x = self.backbone.fc(x)
                
        return x

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Python executable: {sys.executable}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Enhanced Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model_name = 'swin_small_patch4_window7_224'  # Upgraded model
    batch_size = 16  # Smaller batch size for better generalization
    num_epochs = 50  # More epochs for convergence
    learning_rate = 5e-5  # Lower learning rate
    lambda_sac = 2.0  # Higher SAC weight
    num_classes = 31
    
    # Improved confidence schedule
    start_conf = 0.60  # Lower starting point
    end_conf = 0.90    # More reasonable upper bound
    
    # Early stopping
    patience = 150  # Reduced patience for faster experimentation
    
    # Single-source configuration: Amazon → DSLR
    source_domain = 'webcam'
    target_domain = 'amazon'
    
    # Create output directory
    output_dir = f'./improved_results_{source_domain}_to_{target_domain}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load datasets with enhanced transforms
    print("Loading datasets with enhanced transformations...")
    print(f"Source: {source_domain} → Target: {target_domain}")
    
    # Note: You'll need to modify your get_loader function to accept transform parameter
    # For now, assuming get_loader handles transforms internally
    src_train_loader = get_loader(source_domain, batch_size, train=True, domain='source')
    trg_train_loader = get_loader(target_domain, batch_size, train=True, domain='target')
    trg_evaluate_loader = get_loader(target_domain, batch_size, train=False, domain='target')

    # Extract DataLoader if tuple
    if isinstance(src_train_loader, tuple):
        src_train_loader = src_train_loader[0]
    if isinstance(trg_train_loader, tuple):
        trg_train_loader = trg_train_loader[0]
    if isinstance(trg_evaluate_loader, tuple):
        trg_evaluate_loader = trg_evaluate_loader[0]
    
    # Verify number of classes
    actual_classes = len(src_train_loader.dataset.classes)
    print(f"Number of classes: {actual_classes}")
    assert actual_classes == num_classes, f"Expected {num_classes} classes, found {actual_classes}"
    
    # Print dataset sizes
    print(f"Source ({source_domain}) samples: {len(src_train_loader.dataset)}")
    print(f"Target train ({target_domain}) samples: {len(trg_train_loader.dataset)}")
    print(f"Target test ({target_domain}) samples: {len(trg_evaluate_loader.dataset)}")
    
    # Initialize enhanced model
    print(f"Initializing enhanced model: {model_name}")
    model = DomainAdaptationModel(model_name, num_classes=num_classes, pretrained=True)
    model.to(device)
    
    # Print model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function
    criterion_cls = nn.CrossEntropyLoss()
    
    # Enhanced optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=1e-5,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training history
    history = {
        'train_cls_loss': [], 'train_sac_loss': [], 'test_acc': [], 
        'test_loss': [], 'epoch_time': [], 'cumulative_time': [],
        'conf_threshold': [], 'learning_rate': []
    }

    # Early stopping variables
    best_acc = 0.0
    best_epoch = 0
    no_improvement_count = 0
    cumulative_time = 0

    print("\nStarting enhanced single-source domain adaptation training...")
    print(f"Confidence schedule: {start_conf:.2f} → {end_conf:.2f}")
    print(f"Lambda SAC: {lambda_sac}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print("-" * 80)
    
    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()

        # Enhanced confidence threshold with warmup
        if epoch < 5:  # Warmup phase
            current_conf = start_conf * (epoch / 5)
        else:
            progress = (epoch - 5) / max(1, num_epochs - 5)
            current_conf = start_conf + (end_conf - start_conf) * (progress ** 0.5)  # Square root schedule
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Train for one epoch
        train_cls_loss, train_sac_loss = train_epoch(
            model, src_train_loader, trg_train_loader, optimizer, 
            criterion_cls, current_conf, lambda_sac, device, epoch
        )
        
        # Evaluate on target test set
        test_acc, test_loss = evaluate(model, trg_evaluate_loader, device, criterion_cls)
        
        # Update learning rate
        scheduler.step()
        
        # Record time
        epoch_time = time.time() - start_time
        cumulative_time += epoch_time
        
        # Update history
        history['train_cls_loss'].append(train_cls_loss)
        history['train_sac_loss'].append(train_sac_loss)
        history['test_acc'].append(test_acc)
        history['test_loss'].append(test_loss)
        history['epoch_time'].append(epoch_time)
        history['cumulative_time'].append(cumulative_time)
        history['conf_threshold'].append(current_conf)
        history['learning_rate'].append(current_lr)
        
        # Print progress with more details
        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"Conf: {current_conf:.3f} | "
              f"LR: {current_lr:.2e} | "
              f"Time: {epoch_time:5.2f}s | "
              f"Train CLS: {train_cls_loss:.4f} | "
              f"SAC: {train_sac_loss:.4f} | "
              f"Test Acc: {test_acc:5.2f}% | "
              f"Best: {best_acc:5.2f}%")
        
        # Early stopping check
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
            no_improvement_count = 0
            
            # Save best model
            model_path = os.path.join(output_dir, f'best_model_w_a_{model_name}.pth')
            torch.save(model.state_dict(), model_path)
            
            # Save checkpoint
            save_checkpoint(model, optimizer, scheduler, history, epoch, test_acc, current_conf, output_dir, model_name)
            
            print(f"✓ New best accuracy: {best_acc:.2f}% (Epoch {best_epoch})")
        else:
            no_improvement_count += 1
            print(f"✗ No improvement ({no_improvement_count}/{patience})")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, scheduler, history, epoch, test_acc, current_conf, output_dir, f"{model_name}_epoch_{epoch+1}")
            
        # Early stopping
        if no_improvement_count >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
            
        # Print separator every 10 epochs
        if (epoch + 1) % 10 == 0:
            print("-" * 80)
            
    # Final results and analysis
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS: {source_domain} → {target_domain}")
    print(f"{'='*60}")
    print(f"Best Test Accuracy: {best_acc:.2f}% (Epoch {best_epoch})")
    print(f"Final Test Accuracy: {history['test_acc'][-1]:.2f}%")
    print(f"Total Training Time: {cumulative_time:.2f}s ({cumulative_time/60:.2f}min)")
    print(f"Average Epoch Time: {np.mean(history['epoch_time']):.2f}s")
    
    # Save final results
    csv_path = save_history_to_csv(history, f'{source_domain}_to_{target_domain}_{model_name}', output_dir)
    
    # Performance assessment
    print(f"\nPerformance Assessment:")
    if best_acc >= 65.98:
        print(f"🎉 EXCELLENT! Strong domain adaptation performance! (Above baseline)")
    elif best_acc >= 65.0:
        print(f"👍 GOOD! Solid adaptation results! (Approaching baseline)")
    elif best_acc >= 63.10:
        print(f"↗️  IMPROVEMENT! Better than previous 73.10%")
    else:
        print(f"⚠️  Needs improvement. Consider further tuning.")
    
    # Plotting suggestion
    print(f"\nNext steps:")
    print(f"1. Check {csv_path} for detailed training history")
    print(f"2. Analyze learning curves for overfitting/underfitting")
    print(f"3. Consider trying 'swin_base_patch4_window7_224' for even better performance")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"CSV file: {csv_path}")
    
    return history, best_acc

if __name__ == '__main__':
    history, best_accuracy = main()