import sys
import os
import numpy as np
import pandas as pd
from data_load import get_loader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torchvision.models.swin_transformer import Swin_T_Weights
import random
from itertools import cycle
import time

# MMD loss
    
def gaussian_kernel(x, y, sigmas=[0.1, 1.0, 10.0]):
    kernel_sum = 0
    for sigma in sigmas:
        xx = torch.matmul(x, x.t())
        yy = torch.matmul(y, y.t())
        xy = torch.matmul(x, y.t())
        x_sqnorms = torch.diag(xx)
        y_sqnorms = torch.diag(yy)
        x_sqnorms_expand = x_sqnorms.unsqueeze(0).expand_as(xx)
        y_sqnorms_expand = y_sqnorms.unsqueeze(0).expand_as(yy)
        exponent = -2 * xy + x_sqnorms_expand.t() + y_sqnorms_expand
        kernel_sum += torch.exp(-exponent / (2 * sigma ** 2))
    return kernel_sum / len(sigmas)

def mkmmd_loss(x, y):
    x_kernel = gaussian_kernel(x, x)
    y_kernel = gaussian_kernel(y, y)
    xy_kernel = gaussian_kernel(x, y)
    mmd = torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)
    return mmd

def get_swin_t_model(model_name='swin_t', weights='IMAGENET1K_V1', no_classes=20):
    if model_name == 'swin_t':
        model = models.swin_t(weights=Swin_T_Weights.DEFAULT)
        model.head = nn.Linear(model.head.in_features, no_classes)
    else:
        raise ValueError("Only swin_t is supported")
    return model

# train
def train_epoch(model, src_train_loader, trg_train_loader, loss_fn, mmd_weight, optimizer, device):
    
    model.train()

    classification_losses = []
    mmd_losses = []
    accuracies = []
    epoch_start_time = time.time()

    trg_iter = cycle(trg_train_loader)
    
    for X_src, y_src in src_train_loader:
        X_src, y_src = X_src.to(device), y_src.to(device)

        X_trg, _ = next(trg_iter)
        X_trg = X_trg.to(device)

        min_batch_size = min(X_src.size(0), X_trg.size(0))
        if min_batch_size == 0:
            continue
        
        X_src = X_src[:min_batch_size]
        y_src = y_src[:min_batch_size]
        X_trg = X_trg[:min_batch_size]

        y_pred_src = model(X_src)
        y_pred_trg = model(X_trg)

        classification_loss = loss_fn(y_pred_src, y_src)
        
        mmd_loss_val = mkmmd_loss(y_pred_src, y_pred_trg) 
        
        combined_loss = classification_loss + mmd_weight * mmd_loss_val
        
        optimizer.zero_grad()
        combined_loss.backward()
        optimizer.step()
        
        classification_losses.append(classification_loss.item())
        mmd_losses.append(mmd_loss_val.item())

        pred = y_pred_src.argmax(dim=1)
        correct = (pred == y_src).float().sum()
        accuracy = correct / min_batch_size
        accuracies.append(accuracy.item())
        
    avg_classification_loss = np.mean(classification_losses) if classification_losses else 0.0
    avg_mmd_loss = np.mean(mmd_losses) if mmd_losses else 0.0
    avg_accuracy = 100 * np.mean(accuracies) if accuracies else 0.0
    epoch_time = time.time() - epoch_start_time

    return avg_classification_loss, avg_accuracy, avg_mmd_loss, epoch_time

# Evaluation
def evaluation(model, trg_evaluate_loader, loss_fn, device): 
    test_loss = []
    test_acc = []
    model.eval()
    
    with torch.no_grad():
        for X, y in trg_evaluate_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            test_loss.append(loss.item())
            
            batch_acc = ((y_pred.argmax(dim=1) == y).float().mean().item())
            test_acc.append(batch_acc)

    
    avg_test_loss = np.mean(test_loss)
    avg_test_acc = 100 * np.mean(test_acc)
    
    return avg_test_loss, avg_test_acc

def save_history_to_csv(history, model_name, output_dir):
    epochs = list(range(1, len(history['train_combined_loss']) + 1))
    df = pd.DataFrame({
        'Epoch': epochs,
        'Train_Loss': history['train_combined_loss'],
        'Train_Accuracy': history['train_acc'],
        'Test_Loss': history['test_loss'],
        'MMD_Loss': history['mmd_loss'],
        'Test_Accuracy': history['test_acc'],
        'Epoch_Time_s': history['epoch_time'],  
        'Cumulative_Time_s': history['cumulative_time']
    })
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f'{model_name}_mkmmd_a_w_results_{timestamp}.csv')
    
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")
    return csv_path 


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print(f"python executable: {sys.executable}")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_classes = 31
    batch_size = 32
    no_epochs = 50
    learning_rate = 3e-4
    mmd_weight = 0.5
    patience = 100

    output_dir = './results'
    os.makedirs(output_dir, exist_ok=True)
    
    src_train_loader = get_loader(name_dataset='amazon', batch_size=batch_size, train=True)
    trg_train_loader = get_loader(name_dataset='webcam', batch_size=batch_size, train=True)
    trg_evaluate_loader = get_loader(name_dataset='webcam', batch_size=batch_size, train=False)

    print("Analyzing class imbalance in the source dataset:")

    all_source_labels = []
    for _, labels in src_train_loader:
        all_source_labels.extend(labels.tolist())

    all_source_labels = torch.tensor(all_source_labels)
    class_counts = torch.bincount(all_source_labels, minlength=output_classes)
    print(f"Class counts: {class_counts.tolist()}")
    print(f"Number of classes detected: {(class_counts > 0).sum().item()}")

    # Class weights with handling for zero-count classes
    class_counts = class_counts.to(device)
    class_weights = torch.ones(output_classes, device=device)
    nonzero_indices = class_counts > 0
    class_weights[nonzero_indices] = 1.0 / class_counts[nonzero_indices].float()
    class_weights = class_weights / class_weights.sum()
    print(f"Class weights: {class_weights.tolist()}")


    no_classes = len(src_train_loader.dataset.classes)
    print(f"Number of classes: {no_classes}")
    assert no_classes == output_classes, f"Expected {output_classes} classes found {no_classes}"

    model_names = ['swin_t']
    histories = {} 
    for model_name in model_names:
        print(f"\nTraining {model_name} model...")
        model = get_swin_t_model(model_name, no_classes=output_classes)
        model = model.to(device)

        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW([
        {'params': model.head.parameters(), 'lr': learning_rate},
        {'params': [p for p in model.parameters() if id(p) not in [id(param) for param in model.head.parameters()]], 'lr': learning_rate / 10}
            ], weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=no_epochs)

        history = {
            'train_combined_loss': [], 
            'train_acc': [], 
            'test_acc': [], 
            'test_loss': [], 
            'mmd_loss': [],
            'epoch_time': [], 
            'cumulative_time': []
        }
        counter = 0
        best_acc = 0.0
        cumulative_time = 0

        for epoch in range(no_epochs):
            # Get epoch time from train_epoch
            train_loss, train_acc, mmd_loss_val, epoch_time = train_epoch(
                model, src_train_loader, trg_train_loader, loss_fn, mmd_weight, optimizer, device
            )
            test_loss, test_acc = evaluation(model, trg_evaluate_loader, loss_fn, device)
            cumulative_time += epoch_time

            history['train_combined_loss'].append(train_loss + mmd_loss_val)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            history['mmd_loss'].append(mmd_loss_val)  
            history['test_loss'].append(test_loss)
            history['epoch_time'].append(epoch_time)
            history['cumulative_time'].append(cumulative_time)
            
            scheduler.step()

            print(f"{model_name}-Epoch {epoch+1}/{no_epochs}, Train Loss: {train_loss:.4f}, "
                  f"Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, MKMMD Loss: {mmd_loss_val:.4f}, "
                  f"Epoch Time: {epoch_time:.2f}s, Cumulative: {cumulative_time:.2f}s, "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            if test_acc > best_acc:
                best_acc = test_acc
                counter = 0
                torch.save(model.state_dict(), f'MKmmd_a_w_{model_name}.pth')
                print(f"{model_name} - New best accuracy: {best_acc:.2f}%. Model saved.")
            else:
                counter += 1
                print(f"{model_name} - No improvement. Patience counter: {counter}/{patience}")
            
            if counter >= patience:
                print(f"{model_name} - Early stopping triggered after {patience} epochs with no improvement.")
                break
        
        print(f"\n--- {model_name} Timing Analysis ---")
        print(f"Total Training Time: {cumulative_time:.2f} seconds ({cumulative_time/60:.2f} minutes)")
        print(f"Average Time per Epoch: {np.mean(history['epoch_time']):.2f} seconds")
        print(f"Fastest Epoch: {np.min(history['epoch_time']):.2f} seconds")
        print(f"Slowest Epoch: {np.max(history['epoch_time']):.2f} seconds")

        histories[model_name] = history
        save_history_to_csv(history, model_name, output_dir)
        print(f'{model_name} - Training finished. Best Test Accuracy: {best_acc:.2f}%')

    print("\nFinal Answer")
    for model_name in model_names:
        best_acc = max(histories[model_name]['test_acc'])
        print(f"{model_name} - Best Test Accuracy: {best_acc:.2f}%")
    return histories

if __name__ == '__main__':
    histories = main()