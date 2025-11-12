import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_experiment(csv_path):
    """
    Analyzes a single experiment CSV file and creates plots.
    """
    # Load results
    df = pd.read_csv(csv_path)
    
    # Extract model name from filename
    filename = os.path.basename(csv_path)
    model_name = filename.split('_')[0] + '_' + filename.split('_')[1]  # Get 'swin_t'
    
    # Create output directory for plots in the same location as your CSV
    results_dir = os.path.dirname(csv_path)
    plot_dir = os.path.join(results_dir, 'cls_plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # 1. Final Timing Analysis
    total_time = df['Cumulative_Time_s'].iloc[-1]
    avg_epoch_time = df['Epoch_Time_s'].mean()
    best_accuracy = df['Test_Accuracy'].max()
    
    print(f"\n--- {model_name} Timing Analysis ---")
    print(f"Total Training Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Average Time per Epoch: {avg_epoch_time:.2f} seconds")
    print(f"Best Test Accuracy: {best_accuracy:.2f}%")
    
    # 2. Create Accuracy/Loss Curves Plot
    plt.figure(figsize=(12, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(df['Epoch'], df['Train_Accuracy'], 'b-', label='Train Accuracy', linewidth=2)
    plt.plot(df['Epoch'], df['Test_Accuracy'], 'r-', label='Test Accuracy', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{model_name} - Accuracy Curves\nBest Test: {best_accuracy:.2f}%')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(df['Epoch'], df['Train_Loss'], 'b-', label='Train Loss', linewidth=2)
    plt.plot(df['Epoch'], df['Test_Loss'], 'r-', label='Test Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} - Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(plot_dir, f'{model_name}_a_d_training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to {plot_path}")
    
    # 3. Create Timing Analysis Plot
    plt.figure(figsize=(10, 5))
    plt.plot(df['Epoch'], df['Epoch_Time_s'], 'g-o', markersize=4, linewidth=2)
    plt.axhline(y=avg_epoch_time, color='r', linestyle='--', 
                label=f'Average: {avg_epoch_time:.2f}s/epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.title(f'{model_name} - Time per Epoch\nTotal: {total_time/60:.1f} minutes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    timing_path = os.path.join(plot_dir, f'{model_name}_a_d_timing_analysis.png')
    plt.savefig(timing_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved timing analysis to {timing_path}")
    
    return df

if __name__ == '__main__':
    # Use your specific CSV file path
    specific_csv_path = r"C:\\Users\\Admin\\Aruna_jrf_projects\\Datasets\\Office-31\\paper_experiments\\results\\swin_t_office_cls_results_20250913_165252.csv"
    
    # Analyze just this specific file
    print(f"Analyzing specific file: {specific_csv_path}")
    analyze_experiment(specific_csv_path)
    print("\nAnalysis complete!")