import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch
import numpy as np
from torch.utils.data import WeightedRandomSampler
from collections import Counter

def get_class_weights(dataset):
    class_counts = []
    for class_idx in range(len(dataset.classes)):
        count = sum(1 for _, label in dataset.samples if label == class_idx)
        class_counts.append(count)
    
    # Convert to weights (inverse frequency)
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
    weights = weights / weights.sum()  # Normalize
    return weights

def get_balanced_sampler(dataset):
    class_counts = []
    for class_idx in range(len(dataset.classes)):
        count = sum(1 for _, label in dataset.samples if label == class_idx)
        class_counts.append(count)
    
    # Create sample weights
    sample_weights = [1.0 / class_counts[label] for _, label in dataset.samples]
    
    # Create sampler
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    return sampler

def get_loader(name_dataset, batch_size, train=True, domain='source', handle_imbalance=False):
    dataset_paths = {
        'amazon': 'C:\\Users\\Admin\\Aruna_jrf_projects\\Datasets\\Office-31\\amazon',
        'dslr': 'C:\\Users\\Admin\\Aruna_jrf_projects\\Datasets\\Office-31\\dslr',
        'webcam': 'C:\\Users\\Admin\\Aruna_jrf_projects\\Datasets\\Office-31\\webcam'
    }
    
    if train:
        if domain == 'source':  # strong augmentation
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:  # weak augmentation for trg domain
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    else:  # for both src and trg domain for evaluation
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    dataset_path = dataset_paths.get(name_dataset)
    if dataset_path is None:
        raise ValueError(f"Dataset {name_dataset} not found in defined paths")
    
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    
    # Handle imbalance if requested
    if train and handle_imbalance:
        sampler = get_balanced_sampler(dataset)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                          num_workers=4, pin_memory=True, drop_last=train)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=train,
                          num_workers=4, pin_memory=True, drop_last=train)
    
    return loader, dataset  # Return both loader and dataset