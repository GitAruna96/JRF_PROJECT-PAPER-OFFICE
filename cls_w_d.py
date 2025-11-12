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
import time

def get_swin_t_model(model_name='swin_t', weights = 'IMAGENET1K_V1', no_classes=20):
    if model_name == 'swin_t':
        model = models.swin_t(weights=Swin_T_Weights.DEFAULT)
        model.head = nn.Linear(model.head.in_features, no_classes)
    else:
        raise ValueError("Only swin_t is supported")
    return model

# train
def train_epoch(model, src_train_loader, loss_fn, optimizer, device):
    train_loss = []
    train_acc = []
    model.train()

    epoch_start_time = time.time()
    
    for X, y in src_train_loader:
        X, y = X.to(device), y.to(device)
        y_pred_src = model(X)
        classification_loss = loss_fn(y_pred_src, y)
        loss = classification_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss.append(loss.item())
        train_acc.append((y_pred_src.argmax(dim=1) == y).float().mean().item())

    epoch_time = time.time() - epoch_start_time
    
    avg_train_loss = np.mean(train_loss)
    avg_train_acc = 100 * np.mean(train_acc)
    
    return avg_train_loss, avg_train_acc, epoch_time

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
    epochs = list(range(1, len(history['train_cls_loss']) + 1))
    df = pd.DataFrame({
        'Epoch': epochs,
        'Train_Loss': history['train_cls_loss'],
        'Train_Accuracy': history['train_acc'],
        'Test_Loss': history['test_loss'],
        'Test_Accuracy': history['test_acc'],
        'Epoch_Time_s': history['epoch_time'],  
        'Cumulative_Time_s': history['cumulative_time']  
    })
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f'{model_name}_office_cls__w_d_results_{timestamp}.csv')
    
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")
    return csv_path  # Return the path so you can use it later


    
def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print(f"python executable: {sys.executable}")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model_names = ['swin_t']
    output_classes = 31
    batch_size = 32
    no_epochs = 50
    learning_rate = 1e-4
    patience = 50
     
    output_dir = './results'
    os.makedirs(output_dir, exist_ok=True)

    src_train_loader = get_loader(name_dataset = 'webcam', batch_size = batch_size, train=True)
    trg_evaluate_loader = get_loader(name_dataset = 'dslr', batch_size = batch_size, train=False)

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

    histories = {} 
    for model_name in model_names:
        print(f"\nTraining {model_name} model...") 
        model = get_swin_t_model(model_name, no_classes=output_classes)
        model = model.to(device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=no_epochs)


        history = {'train_cls_loss': [], 'train_acc': [], 'test_acc': [], 'test_loss': [], 'epoch_time': [], 'cumulative_time': []}
        counter = 0
        best_acc = 0.0

        cumulative_time = 0

        for epoch in range(no_epochs):

            train_cls_loss, train_acc, epoch_time = train_epoch(model, src_train_loader, loss_fn, optimizer, device)
            test_loss, test_acc = evaluation(model, trg_evaluate_loader, loss_fn, device)
            cumulative_time += epoch_time

            history['train_cls_loss'].append(train_cls_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            history['test_loss'].append(test_loss)
            history['epoch_time'].append(epoch_time)
            history['cumulative_time'].append(cumulative_time)
            
            scheduler.step()

            print(f"{model_name}-Epoch {epoch+1}/{no_epochs},"
                f"Time: {epoch_time:.2f}s (Total: {cumulative_time:.2f}s), "
                f"Train Loss: {train_cls_loss:.4f}, "
                f"Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            if test_acc > best_acc:
                best_acc = test_acc
                counter = 0
                torch.save(model.state_dict(), f'office_cls_w_d_{model_name}.pth')
                print(f"{model_name} - New best accuracy: {best_acc:.2f}%. Model saved.")
            else:
                counter += 1
                print(f"{model_name} - No improvement. Patience counter: {counter}/{patience}")
            
            if counter >= patience:
                print(f"{model_name} - Early stopping triggered after {patience} epochs with no improvement.")
                break

        # FINAL TIMING ANALYSIS R
        print(f"\n--- {model_name} Timing Analysis ---")
        print(f"Total Training Time: {cumulative_time:.2f} seconds ({cumulative_time/60:.2f} minutes)")
        print(f"Average Time per Epoch: {np.mean(history['epoch_time']):.2f} seconds")
        print(f"Fastest Epoch: {np.min(history['epoch_time']):.2f} seconds")
        print(f"Slowest Epoch: {np.max(history['epoch_time']):.2f} seconds")

        histories[model_name] = history
        save_history_to_csv(history, model_name, output_dir)
        print(f'{model_name} - Training finished. Best Test Accuracy: {best_acc:.2f}%')

    print("\n -------------Final Answer----------------")
    for model_name in model_names:
        best_acc = max(histories[model_name]['test_acc'])
        print(f"{model_name} - Best Test Accuracy: {best_acc:.2f}%")
    return histories

if __name__ == '__main__':
    histories = main()















