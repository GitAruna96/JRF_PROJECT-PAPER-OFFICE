import os
import sys
import torch
import pandas as pd
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
import torch.fft as fft
from sac_data_load import get_loader
from timm import create_model
import time

# Spectral Perturbation
def spectral_perturbation(x, noise_scale=0.05, high_freq_ratio=0.7):
    freq = fft.fft2(x)
    freq_shift = fft.fftshift(freq)
    h, w = x.shape[-2:]
    cy, cx = h // 2, w // 2
    radius = int(min(h, w) * high_freq_ratio / 2)
    mask = torch.ones((h, w), device=x.device)
    mask[cy-radius:cy+radius, cx-radius:cx+radius] = 0
    noise = noise_scale * torch.randn_like(freq_shift) * mask
    freq_perturbed = freq_shift + noise
    freq_back = fft.ifftshift(freq_perturbed)
    x_perturbed = fft.ifft2(freq_back).real.clamp(0, 1)
    return x_perturbed

def train_epoch(model, src_train_loader, trg_train_loader, optimizer, 
                criterion_cls, criterion_mse, conf_threshold, lambda_sac, device):
    """Training for single-source domain adaptation"""
    
    model.train()
    total_cls_loss = 0.0
    total_sac_loss = 0.0
    num_batches = 0

    for src_batch, trg_batch in zip(src_train_loader, trg_train_loader):
        src_imgs, src_labels = src_batch[0].to(device), src_batch[1].to(device)
        trg_imgs = trg_batch[0].to(device)

        # Source classification (supervised)
        src_outputs = model(src_imgs)
        cls_loss = criterion_cls(src_outputs, src_labels)
        total_cls_loss += cls_loss.item()

        # Target: SAC anchor (unsupervised domain adaptation)
        trg_outputs = model(trg_imgs)
        trg_probs = torch.softmax(trg_outputs, dim=1)
        conf_scores, _ = torch.max(trg_probs, dim=1)
        anchor_mask = conf_scores > conf_threshold
        anchor_imgs = trg_imgs[anchor_mask]
        anchor_outputs = trg_outputs[anchor_mask]

        sac_loss = torch.tensor(0.0, device=device)
        if anchor_imgs.size(0) > 0:
            perturbed_anchors = spectral_perturbation(anchor_imgs)
            perturbed_outputs = model(perturbed_anchors)

            original_preds = torch.argmax(anchor_outputs, dim=1)
            perturbed_preds = torch.argmax(perturbed_outputs, dim=1)
            consistency_mask = (original_preds == perturbed_preds)
            
            if consistency_mask.any():
                consistent_anchor = anchor_outputs[consistency_mask]
                consistent_perturbed = perturbed_outputs[consistency_mask]
                sac_loss = criterion_mse(consistent_anchor, consistent_perturbed)
                total_sac_loss += sac_loss.item()

        # Total loss = classification loss + domain adaptation loss
        total_loss = cls_loss + lambda_sac * sac_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
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
        'Epoch_Time_s': history['epoch_time'],  
        'Cumulative_Time_s': history['cumulative_time']  
    })
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f'{model_name}_sac_single_a_w_results_{timestamp}.csv')
    
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")
    return csv_path

def save_checkpoint(model, optimizer, history, epoch, test_acc, conf_threshold, output_dir, model_name):
    """Save complete training state"""
    
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
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
    
    filename = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}_{test_acc:.2f}.pth')
    torch.save(checkpoint, filename)
    print(f"✓ Checkpoint saved: {filename}")
    return filename

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print(f"Python executable: {sys.executable}")
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model_name = 'swin_tiny_patch4_window7_224'
    batch_size = 32
    num_epochs = 30
    learning_rate = 1e-4
    lambda_sac = 1.0  # Weight for SAC loss
    num_classes = 31
    patience = 100  # Early stopping patience

    # Single-source configuration: Amazon → DSLR
    source_domain = 'amazon'
    target_domain = 'webcam'
    
    # Reverse confidence schedule
    start_conf = 0.80
    end_conf = 0.95
    
    # Create output directory
    output_dir = f'./results_{source_domain}_to_{target_domain}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load datasets
    print("Loading datasets for single-source domain adaptation...")
    print(f"Source: {source_domain} → Target: {target_domain}")
    
    # Load source and target datasets
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
    
    # Initialize model
    print(f"Initializing model: {model_name}")
    model = create_model(model_name, pretrained=True, num_classes=num_classes)
    model.to(device)
    
    # Loss functions
    criterion_cls = nn.CrossEntropyLoss()  # Classification loss
    criterion_mse = nn.MSELoss()           # SAC consistency loss
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Training history
    history = {
        'train_cls_loss': [], 'train_sac_loss': [], 'test_acc': [], 
        'test_loss': [], 'epoch_time': [], 'cumulative_time': [],
        'conf_threshold': []
    }

    # Early stopping variables
    best_acc = 0.0
    best_epoch = 0
    no_improvement_count = 0
    cumulative_time = 0

    print("Starting single-source domain adaptation training...")
    print(f"Confidence schedule: {start_conf:.2f} → {end_conf:.2f}")
    print(f"Lambda SAC: {lambda_sac}")
    
    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()

        # Calculate current confidence threshold (reverse schedule)
        current_conf = start_conf + (end_conf - start_conf) * (epoch / max(1, num_epochs-1))
        
        # Train for one epoch
        train_cls_loss, train_sac_loss = train_epoch(
            model, src_train_loader, trg_train_loader, optimizer, 
            criterion_cls, criterion_mse, current_conf, lambda_sac, device
        )
        
        # Evaluate on target test set
        test_acc, test_loss = evaluate(model, trg_evaluate_loader, device, criterion_cls)
        
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
        
        # Print progress
        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"Conf: {current_conf:.3f} | "
              f"Time: {epoch_time:5.2f}s | "
              f"Train CLS: {train_cls_loss:.4f} | "
              f"SAC: {train_sac_loss:.4f} | "
              f"Test Acc: {test_acc:5.2f}%")
        
        # Early stopping check
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
            no_improvement_count = 0
            
            # Save best model
            model_path = os.path.join(output_dir, f'best_modelsac_a_w__{model_name}.pth')
            torch.save(model.state_dict(), model_path)
            
            # Save checkpoint
            save_checkpoint(model, optimizer, history, epoch, test_acc, current_conf, output_dir, model_name)
            
            print(f"✓ New best accuracy: {best_acc:.2f}% (Epoch {best_epoch})")
        else:
            no_improvement_count += 1
            print(f"✗ No improvement ({no_improvement_count}/{patience})")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, history, epoch, test_acc, current_conf, output_dir, f"{model_name}_epoch_{epoch+1}")
            
        # Early stopping
        if no_improvement_count >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
            
    # Final results
    print(f"\n{'='*50}")
    print(f"FINAL RESULTS: {source_domain} → {target_domain}")
    print(f"{'='*50}")
    print(f"Best Test Accuracy: {best_acc:.2f}% (Epoch {best_epoch})")
    print(f"Total Training Time: {cumulative_time:.2f}s ({cumulative_time/60:.2f}min)")
    print(f"Average Epoch Time: {np.mean(history['epoch_time']):.2f}s")
    
    # Save final results
    csv_path = save_history_to_csv(history, f'{source_domain}_to_{target_domain}_{model_name}', output_dir)
    
    # Performance assessment
    print(f"\nPerformance Assessment:")
    if best_acc >= 78.06:
        print(f"🎉 EXCELLENT! Strong domain adaptation performance!")
    elif best_acc >= 70.0:
        print(f"👍 GOOD! Solid adaptation results!")
    else:
        print(f"⚠️  Needs improvement. Consider tuning hyperparameters.")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"CSV file: {csv_path}")
    
    return history, best_acc

if __name__ == '__main__':
    history, best_accuracy = main()