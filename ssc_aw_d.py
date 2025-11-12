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
from itertools import cycle
import torchvision.transforms as transforms
import time


def ssc_loss(features_view1, features_view2, temperature=0.05):
    features_view1 = F.normalize(features_view1, dim=1)
    features_view2 = F.normalize(features_view2, dim=1)
    similarity_matrix = torch.mm(features_view1, features_view2.t()) / temperature
    positives = similarity_matrix.diag().view(-1, 1)
    mask = torch.eye(features_view1.size(0), device=features_view1.device, dtype=torch.bool)
    negatives = similarity_matrix[~mask].view(similarity_matrix.size(0), -1)
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.size(0), device=logits.device, dtype=torch.long)
    loss = F.cross_entropy(logits, labels)
    return loss


class SwinSSC(nn.Module):
    def __init__(self, model_name='swin_t', weights=Swin_T_Weights.DEFAULT, no_classes=20):
        super(SwinSSC, self).__init__()
        self.backbone = models.swin_t(weights=weights)
        self.in_features = self.backbone.head.in_features
        self.backbone.head = nn.Identity()
        self.classifier = nn.Linear(self.in_features, no_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, return_features=False):
    # Input should always be 4D: [batch, channels, height, width]
        features = self.backbone(x)
        out = self.classifier(self.dropout(features))
        if return_features:
            return out, features
        else:
            return out


def train_epoch(model, src_loader, trg_loader, loss_fn, optimizer, device, ssc_weight, temperature, cls_weight):
    model.train()
    total_cls_loss = 0
    total_ssc_loss = 0
    total_acc = 0
    num_batches = 0
    
    trg_loader_iter = cycle(trg_loader)
    
    for X_src, y_src in src_loader:
        X_src, y_src = X_src.to(device), y_src.to(device)

        # Get target batch - handle different formats
        batch_data = next(trg_loader_iter)
        
        # Check the format returned by the target loader
        if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
            # This could be either (image, label) or ((view1, view2), label)
            first_item = batch_data[0]
            
            if isinstance(first_item, (list, tuple)) and len(first_item) == 2:
                # Format: ((view1, view2), label)
                X_trg_views, _ = batch_data
                X_trg_view1 = X_trg_views[0].to(device)
                X_trg_view2 = X_trg_views[1].to(device)
            else:
                # Format: (image, label) - create two views from the same image
                X_trg_single, _ = batch_data
                X_trg_view1 = X_trg_single.to(device)
                X_trg_view2 = X_trg_single.to(device)
        else:
            # Fallback: assume it's a single image
            X_trg_view1 = batch_data.to(device)
            X_trg_view2 = batch_data.to(device)
        
        optimizer.zero_grad()

        # 1. SUPERVISED LOSS on SOURCE domain
        y_pred_src, _ = model(X_src, return_features=True)
        classification_loss = loss_fn(y_pred_src, y_src)

        # 2. SELF-SUPERVISED CONTRASTIVE LOSS on TARGET domain
        _, features_view1 = model(X_trg_view1, return_features=True)
        _, features_view2 = model(X_trg_view2, return_features=True)

        ssc_loss_val = ssc_loss(features_view1, features_view2, temperature)

        # Combine Total loss
        loss = cls_weight * classification_loss + ssc_weight * ssc_loss_val 
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_cls_loss += classification_loss.item()
        total_ssc_loss += ssc_loss_val.item() 
        acc = (torch.argmax(y_pred_src, dim=1) == y_src).float().mean().item()
        total_acc += acc
        num_batches += 1

    avg_cls_loss = total_cls_loss / num_batches
    avg_ssc_loss = total_ssc_loss / num_batches
    avg_acc = 100 * (total_acc / num_batches)
    
    return avg_cls_loss, avg_ssc_loss, avg_acc

def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    total_acc = 0
    num_batches = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            
            loss = loss_fn(y_pred, y)
            total_loss += loss.item()
            acc = (torch.argmax(y_pred, dim=1) == y).float().mean().item()
            total_acc += acc
            num_batches += 1
    return total_loss / num_batches, 100 * (total_acc / num_batches)

def save_history_to_csv(history, model_name, output_dir):
    epochs = list(range(1, len(history['train_cls_loss']) + 1))
    df = pd.DataFrame({
        'Epoch': epochs,
        'Train_Loss': history['train_cls_loss'],
        'Train_Accuracy': history['train_acc'],
        'SSC_Loss': history['train_ssc_loss'],
        'Test_Loss': history['test_loss'],
        'Test_Accuracy': history['test_acc'],
        'Epoch_Time_s': history['epoch_time'],
        'Cumulative_Time_s': history['cumulative_time']
    })
    csv_path = os.path.join(output_dir, f'{model_name} office_11_aw_d_ssc_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")


def main():
    print(f"python executable: {sys.executable}")
    print(f"PyTorch version: {torch.__version__}")
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model_name = 'swin_t'
    output_classes = 31
    batch_size = 32
    no_epochs = 50
    learning_rate = 3e-4
    ssc_weight = 1.5
    temperature = 0.07
    cls_weight = 1.0
    weight_decay = 0.1
    
    output_dir = './results'
    os.makedirs(output_dir, exist_ok=True)
    
    # CORRECTED: Use the correct function signature from your data loader
    src_train_loader = get_loader(name_dataset='amazon', batch_size=batch_size, train=True)
    trg_train_loader = get_loader(name_dataset='webcam', batch_size=batch_size, train=True)
    trg_evaluate_loader = get_loader(name_dataset='dslr', batch_size=batch_size, train=False)

    # Debug: Check what each loader returns
    print("Checking source loader format:")
    src_sample = next(iter(src_train_loader))
    print(f"Source loader returns: {type(src_sample)}")
    if isinstance(src_sample, (list, tuple)):
        print(f"Length: {len(src_sample)}")
        for i, item in enumerate(src_sample):
            print(f"Item {i} type: {type(item)}, shape: {item.shape if hasattr(item, 'shape') else 'N/A'}")
    
    print("\nChecking target training loader format:")
    trg_sample = next(iter(trg_train_loader))
    print(f"Target training loader returns: {type(trg_sample)}")
    if isinstance(trg_sample, (list, tuple)):
        print(f"Length: {len(trg_sample)}")
        for i, item in enumerate(trg_sample):
            print(f"Item {i} type: {type(item)}, shape: {item.shape if hasattr(item, 'shape') else 'N/A'}")
    
    print("\nChecking target evaluation loader format:")
    eval_sample = next(iter(trg_evaluate_loader))
    print(f"Target evaluation loader returns: {type(eval_sample)}")
    if isinstance(eval_sample, (list, tuple)):
        print(f"Length: {len(eval_sample)}")
        for i, item in enumerate(eval_sample):
            print(f"Item {i} type: {type(item)}, shape: {item.shape if hasattr(item, 'shape') else 'N/A'}")

    # Class imbalance analysis
    print("Analyzing class imbalance in the source dataset:")
    all_source_labels = []
    for _, labels in src_train_loader:
        all_source_labels.extend(labels.tolist())

    all_source_labels = torch.tensor(all_source_labels)
    class_counts = torch.bincount(all_source_labels, minlength=output_classes)
    print(f"Class counts: {class_counts.tolist()}")
    
    # Class weights
    class_counts = class_counts.to(device)
    class_weights = torch.ones(output_classes, device=device)
    nonzero_indices = class_counts > 0
    class_weights[nonzero_indices] = 1.0 / class_counts[nonzero_indices].float()
    class_weights = class_weights / class_weights.sum()
    
    no_classes = len(src_train_loader.dataset.classes)
    print(f"Number of classes: {no_classes}")
    
    model = SwinSSC(model_name=model_name, weights=Swin_T_Weights.DEFAULT, no_classes=output_classes)
    model = model.to(device)
    
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5, verbose=True, min_lr=1e-6)      
              
    history = {
        'train_cls_loss': [], 'train_acc': [], 'train_ssc_loss': [], 
        'test_acc': [], 'test_loss': [],
        'epoch_time': [], 'cumulative_time': []
    }
    
    best_acc = 0
    cumulative_time = 0

    print("Starting SSC training...")
    
    for epoch in range(no_epochs):
        epoch_start_time = time.time()
        train_cls_loss, train_ssc_loss, train_acc = train_epoch(
            model, src_train_loader, trg_train_loader, loss_fn, optimizer, device, ssc_weight, temperature, cls_weight
        )
        test_loss, test_acc = evaluate(model, trg_evaluate_loader, loss_fn, device)

        epoch_time = time.time() - epoch_start_time
        cumulative_time += epoch_time
        
        # Update history
        history['train_cls_loss'].append(train_cls_loss)
        history['train_acc'].append(train_acc)
        history['train_ssc_loss'].append(train_ssc_loss)
        history['test_acc'].append(test_acc)
        history['test_loss'].append(test_loss)
        history['epoch_time'].append(epoch_time)
        history['cumulative_time'].append(cumulative_time)

        scheduler.step(test_loss)
        
        print(f'Epoch {epoch+1:2d}/{no_epochs}: | Time: {epoch_time:.2f}s | '
              f'CLS Loss: {train_cls_loss:.4f} | SSC Loss: {train_ssc_loss:.4f} | '
              f'Train Acc: {train_acc:6.2f}% | Test Acc: {test_acc:6.2f}%')
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_modeloffice_11_aw_d_ssc.pth')
            print(f"New best accuracy: {best_acc:.2f}%. Model saved.")

    # Final timing analysis
    print(f"\n--- {model_name} Timing Analysis ---")
    print(f"Total Training Time: {cumulative_time:.2f} seconds ({cumulative_time/60:.2f} minutes)")
    print(f"Average Time per Epoch: {np.mean(history['epoch_time']):.2f} seconds")
    print(f"Fastest Epoch: {np.min(history['epoch_time']):.2f} seconds")
    print(f"Slowest Epoch: {np.max(history['epoch_time']):.2f} seconds")

    save_history_to_csv(history, model_name, output_dir)
    print(f'{model_name} - Training finished. Best Test Accuracy: {best_acc:.2f}%')

    return history


if __name__ == "__main__":
    main()
