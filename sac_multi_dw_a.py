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
from collections import Counter

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

def train_epoch_multi_source(model, src_loaders, trg_train_loader, optimizer, 
                           criterion_cls, criterion_mse, conf_threshold, lambda_sac, device):
    
    model.train()
    total_cls_loss = 0.0
    total_sac_loss = 0.0
    num_batches = 0

    # Create iterators for all source loaders
    src_iters = [iter(loader) for loader in src_loaders]
    trg_iter = iter(trg_train_loader)

    for batch_idx in range(len(trg_train_loader)):
        try:
            # Get target batch
            trg_imgs, _ = next(trg_iter)
            trg_imgs = trg_imgs.to(device)
            
            # Get batches from all source domains
            src_imgs_batch = []
            src_labels_batch = []
            
            for src_iter in src_iters:
                try:
                    src_imgs, src_labels = next(src_iter)
                    src_imgs_batch.append(src_imgs)
                    src_labels_batch.append(src_labels)
                except StopIteration:
                    # Reset iterator if exhausted
                    src_idx = src_iters.index(src_iter)
                    src_iters[src_idx] = iter(src_loaders[src_idx])
                    src_imgs, src_labels = next(src_iters[src_idx])
                    src_imgs_batch.append(src_imgs)
                    src_labels_batch.append(src_labels)
            
            # Combine source data
            src_imgs_combined = torch.cat(src_imgs_batch, dim=0).to(device)
            src_labels_combined = torch.cat(src_labels_batch, dim=0).to(device)

            # Source classification
            src_outputs = model(src_imgs_combined)
            cls_loss = criterion_cls(src_outputs, src_labels_combined)
            total_cls_loss += cls_loss.item()

            # Target: SAC anchor
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

            # Total loss
            total_loss = cls_loss + lambda_sac * sac_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            num_batches += 1
            
        except StopIteration:
            break

    avg_cls_loss = total_cls_loss / num_batches if num_batches > 0 else 0.0
    avg_sac_loss = total_sac_loss / num_batches if num_batches > 0 else 0.0

    return avg_cls_loss, avg_sac_loss

def evaluate(model, eval_loader, device, criterion=None):
    
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
    csv_path = os.path.join(output_dir, f'{model_name}_dw_a_sac_predrev_results_{timestamp}.csv')
    
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")
    return csv_path

def save_checkpoint(model, optimizer, history, epoch, test_acc, conf_threshold, output_dir, model_name):
    """Save complete training state including random states"""
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
    
    # Add CUDA state if available
    if torch.cuda.is_available():
        checkpoint['random_state']['cuda'] = torch.cuda.get_rng_state_all()
    
    filename = os.path.join(output_dir, f'checkpoint_dw_a_epoch_{epoch+1}_{test_acc:.2f}.pth')
    torch.save(checkpoint, filename)
    print(f"✓ Complete checkpoint saved: {filename}")
    return filename

def load_checkpoint(checkpoint_path, model, optimizer):
    """Load complete training state"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Restore random states
    random.setstate(checkpoint['random_state']['python'])
    np.random.set_state(checkpoint['random_state']['numpy'])
    torch.set_rng_state(checkpoint['random_state']['torch'])
    
    if torch.cuda.is_available() and 'cuda' in checkpoint['random_state']:
        torch.cuda.set_rng_state_all(checkpoint['random_state']['cuda'])
    
    print(f"Loaded checkpoint for sac_rev_pred from epoch {checkpoint['epoch']} with accuracy {checkpoint['test_accuracy']:.2f}%")
    return checkpoint

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
    lambda_sac = 1.0
    #conf_threshold = 0.85
    num_classes = 31
    patience = 100

    source_domains = ['dslr', 'webcam']  # Your multi-source domains
    target_domain = 'amazon'

    # Reverse confidence schedule
    start_conf = 0.80
    end_conf = 0.95
    
    # Create output directory
    output_dir = './results_multi_source'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load datasets
    print("Loading multi-source datasets...")
    src_train_loaders = []
    for src_domain in source_domains:
        loader, dataset = get_loader(src_domain, batch_size, train=True, domain='source')  # UNPACK
        src_train_loaders.append(loader)
        print(f"Loaded source domain: {src_domain} ({len(dataset)} samples)")  # Use dataset

    trg_train_loader, _ = get_loader(target_domain, batch_size, train=True, domain='target')  # UNPACK, ignore dataset
    trg_evaluate_loader, _ = get_loader(target_domain, batch_size, train=False, domain='target')  # UNPACK, ignore dataset
        
    # Verify number of classes
    actual_classes = len(src_train_loaders[0].dataset.classes)
    print(f"Number of classes: {actual_classes}")
    assert actual_classes == num_classes, f"Expected {num_classes} classes, found {actual_classes}"
    
    # Initialize model, optimizer, and loss functions
    print(f"Initializing model: {model_name}")
    model = create_model(model_name, pretrained=True, num_classes=num_classes)
    model.to(device)
    
    # Use class weights for the first source domain (or combine weights)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()
    
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

    print(f"Starting MULTI-SOURCE training with {len(source_domains)} sources:")
    print(f"Sources: {source_domains}")
    print(f"Target: {target_domain}")
    print(f"Confidence: {start_conf:.2f} → {end_conf:.2f}")
    
    for epoch in range(num_epochs):
        start_time = time.time()

        # Calculate current confidence threshold
        current_conf = start_conf + (end_conf - start_conf) * (epoch / max(1, num_epochs-1))
        
        # Train for one epoch with MULTI-SOURCE
        train_cls_loss, train_sac_loss = train_epoch_multi_source(
            model, src_train_loaders, trg_train_loader, optimizer, 
            criterion_cls, criterion_mse, current_conf, lambda_sac, device
        )
        
        # Evaluate
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
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Conf: {current_conf:.3f}, "
              f"Time: {epoch_time:.2f}s, "
              f"Train CLS: {train_cls_loss:.4f}, SAC: {train_sac_loss:.4f}, "
              f"Test Acc: {test_acc:.2f}%")
        
        # Early stopping check
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
            no_improvement_count = 0
            
            # Save regular model
            torch.save(model.state_dict(), os.path.join(output_dir, f'multi_source_sac_rev_pred_dw_a_{model_name}.pth'))
            
            # Save complete checkpoint
            save_checkpoint(model, optimizer, history, epoch, test_acc, current_conf, output_dir, model_name)
            
            print(f"✓ New best accuracy: {best_acc:.2f}%. Model and checkpoint saved.")
        else:
            no_improvement_count += 1
            print(f"✗ No improvement. Counter: {no_improvement_count}/{patience}")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, history, epoch, test_acc, current_conf, output_dir, f"{model_name}_epoch_{epoch+1}")
            
    # Final results
    print(f"\n--- Final Multi-Source Results ---")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print(f"Total Training Time: {cumulative_time:.2f} seconds ({cumulative_time/60:.2f} minutes)")
    print(f"Average Time per Epoch: {np.mean(history['epoch_time']):.2f} seconds")
    
    # Save results
    save_history_to_csv(history, f'multi_source_{model_name}_dw_a_sac_rev_pred', output_dir)
    
    # Compare with baseline
    baseline_acc = 82.10  # You might want to adjust this for multi-source
    improvement = best_acc - baseline_acc
    print(f"\nComparison with Baseline:")
    print(f"Baseline (single-source): {baseline_acc:.2f}%")
    print(f"Your Multi-Source SAC AD method: {best_acc:.2f}%")
    print(f"Difference: {improvement:+.2f}%")
    
    if best_acc > baseline_acc:
        print("🎉 CONGRATULATIONS! Multi-source approach beat the baseline!")
    else:
        print("⚠️  Let's analyze the multi-source performance.")
    
    return history

if __name__ == '__main__':
    history = main()