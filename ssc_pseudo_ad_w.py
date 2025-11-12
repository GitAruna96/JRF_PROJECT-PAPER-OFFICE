import sys
import os
import numpy as np
import pandas as pd
from data_load_ssc import get_loader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torchvision.models.swin_transformer import Swin_T_Weights
from itertools import cycle
import time
import itertools

def ssc_loss(features_view1, features_view2, temperature=0.07):
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
        features = self.backbone(x)
        out = self.classifier(self.dropout(features))
        if return_features:
            return out, features
        else:
            return out

def generate_pseudo_labels(model, loader, device, threshold):
    model.eval()
    pseudo_list = []
    with torch.no_grad():
        for (X_view1, _), _ in loader:
            X_view1 = X_view1.to(device)
            logits = model(X_view1)
            probs = F.softmax(logits, dim=1)
            conf, preds = torch.max(probs, dim=1)
            pseudo_y = torch.full_like(preds, -1)
            mask = conf > threshold
            pseudo_y[mask] = preds[mask]
            pseudo_list.append(pseudo_y)
    model.train()
    return pseudo_list

def train_epoch(model, src_loader, trg_loader, loss_fn, optimizer, device, cls_weight, ssc_weight, temperature, pseudo_weight, pseudo_batches=None):
    model.train()
    total_cls_loss = total_ssc_loss = total_acc = total_pseudo_loss = 0
    num_batches = 0
    trg_loader_iter = cycle(trg_loader)
    
    # Create iterator for pseudo batches if they exist
    pseudo_iter = cycle(pseudo_batches) if pseudo_batches else None
    
    for X_src, y_src in src_loader:
        X_src, y_src = X_src.to(device), y_src.to(device)
        (X_trg_view1, X_trg_view2), _ = next(trg_loader_iter)
        X_trg_view1, X_trg_view2 = X_trg_view1.to(device), X_trg_view2.to(device)
        optimizer.zero_grad()

        # 1. Supervised loss on source
        y_pred_src, _ = model(X_src, return_features=True)
        classification_loss = loss_fn(y_pred_src, y_src)
        
        # 2. SSC loss on target
        _, features_view1 = model(X_trg_view1, return_features=True)
        _, features_view2 = model(X_trg_view2, return_features=True)
        ssc_loss_val = ssc_loss(features_view1, features_view2, temperature)
        
        # 3. Pseudo-labeling loss
        pseudo_loss_val = 0
        if pseudo_iter is not None:
            pseudo_y = next(pseudo_iter).to(device)
            mask = pseudo_y != -1
            if mask.sum() > 0:
                trg_logits, _ = model(X_trg_view1, return_features=True)
                pseudo_loss_val = loss_fn(trg_logits[mask], pseudo_y[mask])

        # Total loss 
        loss = (cls_weight * classification_loss + 
                ssc_weight * ssc_loss_val + 
                pseudo_weight * pseudo_loss_val)
        
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_cls_loss += classification_loss.item()
        total_ssc_loss += ssc_loss_val.item()
        total_pseudo_loss += pseudo_loss_val if isinstance(pseudo_loss_val, (int, float)) else pseudo_loss_val.item()
        
        acc = (torch.argmax(y_pred_src, dim=1) == y_src).float().mean().item()
        total_acc += acc
        num_batches += 1

    # Calculate averages
    avg_cls_loss = total_cls_loss / num_batches
    avg_ssc_loss = total_ssc_loss / num_batches
    avg_pseudo_loss = total_pseudo_loss / num_batches
    avg_acc = 100 * (total_acc / num_batches)
    
    return avg_cls_loss, avg_ssc_loss, avg_acc, avg_pseudo_loss

def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = total_acc = num_batches = 0
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
        'Pseudo_Loss': history['train_pseudo_loss'],
        'Test_Loss': history['test_loss'],
        'Test_Accuracy': history['test_acc'],
        'Epoch_Time_s': history['epoch_time'],
        'Cumulative_Time_s': history['cumulative_time']
    })
    csv_path = os.path.join(output_dir, f'{model_name}_office_ssc_pseudo_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")

def main():
    torch.cuda.empty_cache()
    print(f"python executable: {sys.executable}")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    model_name = 'swin_t'  
    output_classes = 31
    batch_size = 32
    no_epochs = 50
    learning_rate = 1e-4
    ssc_weight = 1.5
    temperature = 0.07
    weight_decay = 0.01
    cls_weight = 1.0
    warmup_epochs = 5
    pseudo_threshold = 0.95
    pseudo_weight = 0.5
    #patience = 10

    output_dir = './results'
    os.makedirs(output_dir, exist_ok=True)

    # Data loaders
    src_train_loader = get_loader(name_dataset='amazon', batch_size=batch_size, train=True,is_target=False)
    trg_train_loader = get_loader(name_dataset='dslr', batch_size=batch_size, train=True,is_target=True)
    trg_evaluate_loader = get_loader(name_dataset='webcam', batch_size=batch_size, train=False,is_target=False)
    no_classes = len(src_train_loader.dataset.classes)
    print(f"Number of classes: {no_classes}")
    assert no_classes == output_classes, f"Expected {output_classes} classes, found {no_classes}"

    # Model setup
    model = SwinSSC(model_name=model_name, weights=Swin_T_Weights.DEFAULT, no_classes=output_classes)
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    
    # Optimizer with differential learning rates
    backbone_params = model.backbone.parameters()
    head_params = list(model.classifier.parameters()) + list(model.dropout.parameters())
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': learning_rate / 10},
        {'params': head_params, 'lr': learning_rate}
    ], weight_decay=weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=no_epochs)

    # Training history
    history = {
        'train_cls_loss': [], 'train_acc': [], 'train_ssc_loss': [], 
        'train_pseudo_loss': [], 'test_acc': [], 'test_loss': [],
        'epoch_time': [], 'cumulative_time': []
    }
    
    best_acc = 0
    cumulative_time = 0
    counter = 0  # Initialize early stopping counter

    print("Starting SSC + Pseudo Labeling training...")
    
    for epoch in range(no_epochs):
        epoch_start_time = time.time()
        
        # Generate pseudo labels after warmup
        current_pseudo_threshold = pseudo_threshold - (epoch / no_epochs) * 0.1
        pseudo_batches = generate_pseudo_labels(model, trg_train_loader, device, current_pseudo_threshold) if epoch >= warmup_epochs else None
        
        # Training epoch
        train_cls_loss, train_ssc_loss, train_acc, train_pseudo_loss = train_epoch(
            model=model, src_loader=src_train_loader, trg_loader=trg_train_loader, 
            loss_fn=loss_fn, optimizer=optimizer, device=device, 
            cls_weight=cls_weight, ssc_weight=ssc_weight, temperature=temperature, 
            pseudo_weight=pseudo_weight, pseudo_batches=pseudo_batches
        )
        
        # Evaluation
        test_loss, test_acc = evaluate(model, trg_evaluate_loader, loss_fn, device)
        
        # Timing
        epoch_time = time.time() - epoch_start_time
        cumulative_time += epoch_time
        
        # Update history
        history['train_cls_loss'].append(train_cls_loss)
        history['train_acc'].append(train_acc)
        history['train_ssc_loss'].append(train_ssc_loss)
        history['train_pseudo_loss'].append(train_pseudo_loss)
        history['test_acc'].append(test_acc)
        history['test_loss'].append(test_loss)
        history['epoch_time'].append(epoch_time)
        history['cumulative_time'].append(cumulative_time)

        scheduler.step()
        
        print(f'Epoch {epoch+1:2d}/{no_epochs}: | Time: {epoch_time:.2f}s | '
              f'CLS Loss: {train_cls_loss:.4f} | SSC Loss: {train_ssc_loss:.4f} | '
              f'PSEUDO Loss: {train_pseudo_loss:.4f} | '
              f'Train Acc: {train_acc:6.2f}% | Test Acc: {test_acc:6.2f}%')
        
        # Early stopping logic
        if test_acc > best_acc:
            best_acc = test_acc
            counter = 0  # Reset counter
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_office_swin_ssc_pseudo_model.pth'))
            print(f"New best model saved with accuracy: {best_acc:.2f}%")
        '''    
        else:
            counter += 1
            print(f"No improvement. Patience counter: {counter}/{patience}")
        
        if counter >= patience:
            print(f"Early stopping triggered after {patience} epochs with no improvement.")
            break
        '''
    # Final timing analysis
    print(f"\n--- {model_name} Timing Analysis ---")
    print(f"Total Training Time: {cumulative_time:.2f} seconds ({cumulative_time/60:.2f} minutes)")
    print(f"Average Time per Epoch: {np.mean(history['epoch_time']):.2f} seconds")
    print(f"Fastest Epoch: {np.min(history['epoch_time']):.2f} seconds")
    print(f"Slowest Epoch: {np.max(history['epoch_time']):.2f} seconds")

    # Save results
    save_history_to_csv(history, model_name, output_dir)
    print(f'{model_name} - Training finished. Best Test Accuracy: {best_acc:.2f}%')

    return history

if __name__ == '__main__':
    history = main()