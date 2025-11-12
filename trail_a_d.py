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
from torchvision import transforms

# -----------------------
# Spectral + augmentation perturbation
# -----------------------
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
'''
# Image augmentation for SAC
aug_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
    transforms.RandomGrayscale(0.1),
])
'''
# -----------------------
# Training epoch
# -----------------------
def train_epoch(model, src_loader, trg_loader, optimizer, criterion_cls, criterion_mse, conf_threshold, lambda_sac, device):
    model.train()
    total_cls_loss = 0.0
    total_sac_loss = 0.0
    num_batches = 0

    for src_batch, trg_batch in zip(src_loader, trg_loader):
        src_imgs, src_labels = src_batch[0].to(device), src_batch[1].to(device)
        trg_imgs = trg_batch[0].to(device)

        # -----------------------
        # Source supervised loss
        # -----------------------
        src_outputs = model(src_imgs)
        cls_loss = criterion_cls(src_outputs, src_labels)

        # -----------------------
        # Target SAC loss
        # -----------------------
        trg_outputs = model(trg_imgs)
        trg_probs = torch.softmax(trg_outputs, dim=1)
        conf_scores, _ = torch.max(trg_probs, dim=1)

        # Multi-view perturbation + augmentation
        sac_loss = 0.0
        if trg_imgs.size(0) > 0:
            sac_losses = []
            for noise_scale in [0.03, 0.05, 0.07]:
                perturbed_imgs = spectral_perturbation(trg_imgs, noise_scale=noise_scale)
                perturbed_imgs = torch.stack([aug_transforms(img) for img in perturbed_imgs])
                perturbed_outputs = model(perturbed_imgs)
                # Weighted by confidence
                sac_losses.append(((trg_outputs - perturbed_outputs)**2).mean(dim=1) * conf_scores)
            sac_loss = torch.mean(torch.stack(sac_losses))
            total_sac_loss += sac_loss.item()

        # -----------------------
        # Total loss and optimization
        # -----------------------
        total_loss = cls_loss + lambda_sac * sac_loss
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_cls_loss += cls_loss.item()
        num_batches += 1

    avg_cls_loss = total_cls_loss / num_batches
    avg_sac_loss = total_sac_loss / num_batches
    return avg_cls_loss, avg_sac_loss

# -----------------------
# Evaluation
# -----------------------
def evaluate(model, loader, device, criterion=None):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0 if criterion else None
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if criterion:
                total_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(loader) if criterion else None
    return accuracy, avg_loss

# -----------------------
# CSV & checkpoint helpers
# -----------------------
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
        'Cumulative_Time_s': history['cumulative_time'],
    })
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f'{model_name}_sac_a_d_results_{timestamp}.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")
    return csv_path

def save_checkpoint(model, optimizer, history, epoch, test_acc, conf_threshold, output_dir, model_name):
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

# -----------------------
# Main training
# -----------------------
def main():
    # Random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # -----------------------
    # Config
    # -----------------------
    model_name = 'swin_tiny_patch4_window7_224'
    batch_size = 32
    num_epochs = 30
    learning_rate = 1e-4
    lambda_sac = 0.5
    num_classes = 31
    patience = 20

    source_domain = 'amazon'
    target_domain = 'dslr'
    start_conf = 0.6
    end_conf = 0.85

    output_dir = f'./results_{source_domain}_to_{target_domain}'
    os.makedirs(output_dir, exist_ok=True)

    # -----------------------
    # Load datasets
    # -----------------------
    src_train_loader = get_loader(source_domain, batch_size, train=True, domain='source')
    trg_train_loader = get_loader(target_domain, batch_size, train=True, domain='target')
    trg_eval_loader = get_loader(target_domain, batch_size, train=False, domain='target')

    if isinstance(src_train_loader, tuple):
        src_train_loader = src_train_loader[0]
    if isinstance(trg_train_loader, tuple):
        trg_train_loader = trg_train_loader[0]
    if isinstance(trg_eval_loader, tuple):
        trg_eval_loader = trg_eval_loader[0]

    # -----------------------
    # Model
    # -----------------------
    model = create_model(model_name, pretrained=True, num_classes=num_classes)
    # Bottleneck head
    model.head = nn.Sequential(
        nn.Linear(model.head.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    model.to(device)

    # Freeze early layers
    for name, param in model.named_parameters():
        if "layers.0" in name or "layers.1" in name:
            param.requires_grad = False

    # -----------------------
    # Loss + optimizer
    # -----------------------
    criterion_cls = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    history = {
        'train_cls_loss': [], 'train_sac_loss': [], 'test_acc': [],
        'test_loss': [], 'epoch_time': [], 'cumulative_time': [],
        'conf_threshold': []
    }

    best_acc = 0.0
    best_epoch = 0
    no_improvement = 0
    cumulative_time = 0

    # -----------------------
    # Training loop
    # -----------------------
    for epoch in range(num_epochs):
        start_time = time.time()
        # Cosine confidence schedule
        current_conf = end_conf - (end_conf - start_conf) * 0.5 * (1 + np.cos(np.pi * epoch / num_epochs))

        # Unfreeze layers gradually after 5 epochs
        if epoch == 5:
            for name, param in model.named_parameters():
                param.requires_grad = True

        train_cls_loss, train_sac_loss = train_epoch(
            model, src_train_loader, trg_train_loader, optimizer,
            criterion_cls, criterion_mse, current_conf, lambda_sac, device
        )

        test_acc, test_loss = evaluate(model, trg_eval_loader, device, criterion_cls)
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

        print(f"Epoch {epoch+1}/{num_epochs} | Conf: {current_conf:.3f} | "
              f"Train CLS: {train_cls_loss:.4f} | SAC: {train_sac_loss:.4f} | "
              f"Test Acc: {test_acc:.2f}% | Epoch Time: {epoch_time:.2f}s")

        # Early stopping & save best
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
            no_improvement = 0
            torch.save(model.state_dict(), os.path.join(output_dir, f'trail_a_d best_model_{model_name}.pth'))
            save_checkpoint(model, optimizer, history, epoch, test_acc, current_conf, output_dir, model_name)
        else:
            no_improvement += 1

        if no_improvement >= patience:
            print("Early stopping triggered!")
            break

    # Final results
    print(f"\nBest Test Accuracy: {best_acc:.2f}% (Epoch {best_epoch})")
    csv_path = save_history_to_csv(history, f'{source_domain}_to_{target_domain}_{model_name}', output_dir)
    print(f"CSV saved at: {csv_path}")

    return history, best_acc

if __name__ == '__main__':
    history, best_accuracy = main()
