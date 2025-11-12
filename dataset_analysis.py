import torchvision.datasets as datasets
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

def analyze_class_distribution(name_dataset):
    """Analyze class distribution for a given dataset"""
    
    dataset_paths = {
        'amazon': 'C:\\Users\\Admin\\Aruna_jrf_projects\\Datasets\\Office-31\\amazon',
        'dslr': 'C:\\Users\\Admin\\Aruna_jrf_projects\\Datasets\\Office-31\\dslr',
        'webcam': 'C:\\Users\\Admin\\Aruna_jrf_projects\\Datasets\\Office-31\\webcam'
    }
    
    dataset_path = dataset_paths.get(name_dataset)
    if dataset_path is None:
        raise ValueError(f"Dataset {name_dataset} not found in defined paths")
    
    # Load dataset without transformations for analysis
    dataset = datasets.ImageFolder(root=dataset_path)
    
    # Get class labels
    class_labels = [label for _, label in dataset.samples]
    
    # Count occurrences of each class
    class_counts = Counter(class_labels)
    
    # Get class names
    class_names = dataset.classes
    
    # Print results
    print(f"\n=== Class Distribution Analysis for {name_dataset} ===")
    print(f"Total samples: {len(dataset)}")
    print(f"Number of classes: {len(class_names)}")
    print("\nClass distribution:")
    
    for class_idx, count in class_counts.items():
        class_name = class_names[class_idx]
        percentage = (count / len(dataset)) * 100
        print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
    
    # Calculate imbalance metrics
    counts = list(class_counts.values())
    max_count = max(counts)
    min_count = min(counts)
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    print(f"\nImbalance ratio: {imbalance_ratio:.2f}")
    print(f"Max class size: {max_count}")
    print(f"Min class size: {min_count}")
    
    # Plot distribution
    plot_class_distribution(class_counts, class_names, name_dataset)
    
    return class_counts, class_names

def plot_class_distribution(class_counts, class_names, dataset_name):
    """Plot class distribution as a bar chart"""
    
    # Prepare data for plotting
    classes = [class_names[idx] for idx in class_counts.keys()]
    counts = list(class_counts.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(classes)), counts)
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.title(f'Class Distribution - {dataset_name}')
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_class_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

# Analyze all datasets
if __name__ == "__main__":
    datasets_to_analyze = ['amazon', 'dslr', 'webcam']
    
    for dataset_name in datasets_to_analyze:
        try:
            analyze_class_distribution(dataset_name)
        except Exception as e:
            print(f"Error analyzing {dataset_name}: {e}")