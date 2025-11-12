import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import os

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_training_results(csv_path):
    """Create comprehensive visualization of training results"""
    
    # Read the results
    df = pd.read_csv(csv_path)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Accuracy Progress Plot
    plt.subplot(3, 3, 1)
    plt.plot(df['Epoch'], df['Test_Accuracy'], 'b-', linewidth=2.5, label='Test Accuracy')
    plt.axhline(y=94.58, color='r', linestyle='--', linewidth=2, label='Best Accuracy (94.58%)')
    plt.axvline(x=15, color='g', linestyle='--', linewidth=1.5, alpha=0.7, label='Best Epoch (15)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy Progress\nAmazon → DSLR Domain Adaptation', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(70, 100)
    
    # 2. Loss Curves
    plt.subplot(3, 3, 2)
    plt.plot(df['Epoch'], df['Train_CLS_Loss'], 'orange', linewidth=2, label='Classification Loss')
    plt.plot(df['Epoch'], df['Train_SAC_Loss'], 'green', linewidth=2, label='SAC Loss')
    plt.plot(df['Epoch'], df['Test_Loss'], 'red', linewidth=2, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss Curves', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale for better visualization
    
    # 3. Confidence Threshold vs Accuracy
    plt.subplot(3, 3, 3)
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    line1 = ax1.plot(df['Epoch'], df['Conf_Threshold'], 'purple', linewidth=3, label='Confidence Threshold')
    line2 = ax2.plot(df['Epoch'], df['Test_Accuracy'], 'blue', linewidth=2, label='Test Accuracy', alpha=0.7)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Confidence Threshold', color='purple')
    ax2.set_ylabel('Test Accuracy (%)', color='blue')
    ax1.tick_params(axis='y', labelcolor='purple')
    ax2.tick_params(axis='y', labelcolor='blue')
    plt.title('Confidence Threshold vs Accuracy', fontsize=14, fontweight='bold')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 4. Cumulative Training Time
    plt.subplot(3, 3, 4)
    plt.plot(df['Epoch'], df['Cumulative_Time_s']/60, 'brown', linewidth=2.5)
    plt.xlabel('Epoch')
    plt.ylabel('Cumulative Time (minutes)')
    plt.title('Cumulative Training Time\nTotal: 24.13 minutes', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.fill_between(df['Epoch'], df['Cumulative_Time_s']/60, alpha=0.3, color='brown')
    
    # 5. Epoch-wise Time Distribution
    plt.subplot(3, 3, 5)
    plt.bar(df['Epoch'], df['Epoch_Time_s'], color='teal', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Time per Epoch (seconds)')
    plt.title(f'Epoch Training Time\nAverage: {df["Epoch_Time_s"].mean():.2f}s', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 6. Accuracy Distribution (Box plot style)
    plt.subplot(3, 3, 6)
    # Create a "violin plot" style distribution
    accuracy_data = df['Test_Accuracy']
    parts = plt.violinplot([accuracy_data], showmeans=True, showmedians=True)
    
    # Customize colors
    for pc in parts['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_alpha(0.6)
    
    plt.axhline(y=94.58, color='red', linestyle='--', linewidth=2, label='Best Accuracy')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Distribution\nMin: {:.2f}%, Max: {:.2f}%'.format(
        accuracy_data.min(), accuracy_data.max()), fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. Loss vs Accuracy Scatter
    plt.subplot(3, 3, 7)
    scatter = plt.scatter(df['Train_CLS_Loss'], df['Test_Accuracy'], 
                         c=df['Epoch'], cmap='viridis', s=60, alpha=0.7)
    plt.xlabel('Classification Loss')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Loss vs Accuracy Correlation', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='Epoch')
    plt.grid(True, alpha=0.3)
    
    # 8. Convergence Analysis
    plt.subplot(3, 3, 8)
    # Calculate accuracy improvement from previous epoch
    improvement = df['Test_Accuracy'].diff().fillna(0)
    plt.plot(df['Epoch'], improvement, 'orange', linewidth=2, marker='o', markersize=4)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy Improvement (%)')
    plt.title('Epoch-wise Accuracy Improvement', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 9. Performance Summary
    plt.subplot(3, 3, 9)
    # Create a summary table-like plot
    metrics = ['Best Accuracy', 'Final Accuracy', 'Training Time', 'Best Epoch']
    values = [94.58, 93.57, '24.13 min', 15]
    colors = ['green', 'lightgreen', 'orange', 'lightblue']
    
    bars = plt.bar(metrics, [94.58, 93.57, 10, 15], color=colors, alpha=0.7)
    plt.ylabel('Value')
    plt.title('Performance Summary', fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        if i < 2:  # Accuracy values
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{val}%', ha='center', va='bottom', fontweight='bold')
        else:  # Other values
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    str(val), ha='center', va='bottom', fontweight='bold')
    
    plt.ylim(0, 110)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Spectral Anchor Consistency (SAC) - Amazon→DSLR Domain Adaptation Results\n'
                f'Best Accuracy: 94.58% (Epoch 15)', 
                fontsize=16, fontweight='bold', y=1.02)
    
    # Save the plot
    plot_filename = csv_path.replace('.csv', '_analysis.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    return plot_filename

def create_comparison_plot(csv_path):
    """Create comparison with baseline performance"""
    
    df = pd.read_csv(csv_path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left: Accuracy comparison with baseline
    baseline_acc = 73.10  # Your previous result
    current_best = df['Test_Accuracy'].max()
    
    categories = ['Previous Result', 'SAC Improved']
    accuracies = [baseline_acc, current_best]
    colors = ['red', 'green']
    
    bars = ax1.bar(categories, accuracies, color=colors, alpha=0.7, width=0.6)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Performance Improvement\nSpectral Anchor Consistency vs Baseline', 
                 fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # Add improvement percentage
    improvement = ((current_best - baseline_acc) / baseline_acc) * 100
    ax1.text(0.5, 85, f'+{improvement:.1f}% Improvement', 
            ha='center', va='center', fontsize=16, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # Right: Training progress with milestones
    ax2.plot(df['Epoch'], df['Test_Accuracy'], 'b-', linewidth=3, label='Test Accuracy')
    ax2.axhline(y=baseline_acc, color='red', linestyle='--', linewidth=2, 
               label=f'Baseline ({baseline_acc}%)')
    ax2.axhline(y=current_best, color='green', linestyle='--', linewidth=2, 
               label=f'Best SAC ({current_best}%)')
    
    # Mark key epochs
    key_epochs = [5, 10, 15, 20, 25]
    for epoch in key_epochs:
        if epoch <= len(df):
            acc = df.loc[df['Epoch'] == epoch, 'Test_Accuracy'].values[0]
            ax2.plot(epoch, acc, 'ro', markersize=8)
            ax2.annotate(f'{acc:.1f}%', (epoch, acc), 
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training Progress with Milestones', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(70, 100)
    
    plt.tight_layout()
    plt.suptitle('SAC Domain Adaptation: Dramatic Improvement Achieved', 
                fontsize=16, fontweight='bold', y=1.02)
    
    comp_filename = csv_path.replace('.csv', '_comparison.png')
    plt.savefig(comp_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    return comp_filename

# Main execution
if __name__ == "__main__":
    csv_path = "./improved_results_amazon_to_dslr/amazon_to_dslr_swin_small_patch4_window7_224_improved_results_20250923_101958.csv"
    
    try:
        # Plot comprehensive results
        plot_file = plot_training_results(csv_path)
        print(f"✓ Comprehensive analysis plot saved: {plot_file}")
        
        # Plot comparison with baseline
        comp_file = create_comparison_plot(csv_path)
        print(f"✓ Comparison plot saved: {comp_file}")
        
        # Print summary statistics
        df = pd.read_csv(csv_path)
        print(f"\n📊 PERFORMANCE SUMMARY:")
        print(f"   Best Accuracy: {df['Test_Accuracy'].max():.2f}%")
        print(f"   Final Accuracy: {df['Test_Accuracy'].iloc[-1]:.2f}%")
        print(f"   Average Accuracy: {df['Test_Accuracy'].mean():.2f}%")
        print(f"   Standard Deviation: {df['Test_Accuracy'].std():.2f}%")
        print(f"   Training Epochs: {len(df)}")
        print(f"   Average Epoch Time: {df['Epoch_Time_s'].mean():.2f}s")
        print(f"   Total Training: {df['Cumulative_Time_s'].iloc[-1]/60:.2f} min")
        
    except FileNotFoundError:
        print("❌ CSV file not found. Please check the path.")
    except Exception as e:
        print(f"❌ Error occurred: {e}")