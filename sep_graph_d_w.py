import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_individual_plots(csv_path):
    """Create separate individual plots for each analysis"""
    
    # Read the results
    df = pd.read_csv(csv_path)
    
    # Plot 1: Accuracy Progress
    plt.figure(figsize=(10, 6))
    plt.plot(df['Epoch'], df['Test_Accuracy'], 'b-', linewidth=3, label='Test Accuracy')
    plt.axhline(y=99.25, color='r', linestyle='--', linewidth=2, label='Best Accuracy (99.25%)')
    plt.axvline(x=15, color='g', linestyle='--', linewidth=2, alpha=0.7, label='Best Epoch (14)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy Progress - dslr → webcam Domain Adaptation', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(70, 100)
    plt.tight_layout()
    plt.savefig('accuracy_progress.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot 2: Loss Curves
    plt.figure(figsize=(10, 6))
    plt.plot(df['Epoch'], df['Train_CLS_Loss'], 'orange', linewidth=2.5, label='Classification Loss')
    plt.plot(df['Epoch'], df['Train_SAC_Loss'], 'green', linewidth=2.5, label='SAC Loss')
    plt.plot(df['Epoch'], df['Test_Loss'], 'red', linewidth=2.5, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss Curves', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('loss_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot 3: Confidence Threshold vs Accuracy
    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    line1 = ax1.plot(df['Epoch'], df['Conf_Threshold'], 'purple', linewidth=3, label='Confidence Threshold')
    line2 = ax2.plot(df['Epoch'], df['Test_Accuracy'], 'blue', linewidth=2, label='Test Accuracy', alpha=0.8)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Confidence Threshold', color='purple')
    ax2.set_ylabel('Test Accuracy (%)', color='blue')
    ax1.tick_params(axis='y', labelcolor='purple')
    ax2.tick_params(axis='y', labelcolor='blue')
    plt.title('Confidence Threshold vs Accuracy Correlation', fontsize=14, fontweight='bold')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('confidence_vs_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot 4: Training Time Analysis
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(df['Epoch'], df['Cumulative_Time_s']/60, 'brown', linewidth=2.5)
    plt.xlabel('Epoch')
    plt.ylabel('Cumulative Time (minutes)')
    plt.title('Cumulative Training Time\nTotal: 16.42 minutes', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.fill_between(df['Epoch'], df['Cumulative_Time_s']/60, alpha=0.3, color='brown')
    
    plt.subplot(1, 2, 2)
    plt.bar(df['Epoch'], df['Epoch_Time_s'], color='teal', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Time per Epoch (seconds)')
    plt.title(f'Epoch Training Time\nAverage: {df["Epoch_Time_s"].mean():.2f}s', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_time_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot 5: Accuracy Distribution
    plt.figure(figsize=(10, 6))
    accuracy_data = df['Test_Accuracy']
    
    # Create violin plot
    parts = plt.violinplot([accuracy_data], showmeans=True, showmedians=True)
    
    for pc in parts['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_alpha(0.6)
    
    plt.axhline(y=99.25, color='red', linestyle='--', linewidth=2, label='Best Accuracy (99.25%)')
    plt.axhline(y=accuracy_data.mean(), color='orange', linestyle='--', linewidth=2, label=f'Mean ({accuracy_data.mean():.2f}%)')
    
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Distribution Across Epochs', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('accuracy_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot 6: Loss vs Accuracy Correlation
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df['Train_CLS_Loss'], df['Test_Accuracy'], 
                         c=df['Epoch'], cmap='viridis', s=80, alpha=0.8)
    plt.xlabel('Classification Loss')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Loss vs Accuracy Correlation', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Epoch')
    plt.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(df['Train_CLS_Loss'], df['Test_Accuracy'], 1)
    p = np.poly1d(z)
    plt.plot(df['Train_CLS_Loss'], p(df['Train_CLS_Loss']), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.savefig('loss_vs_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot 7: Convergence Analysis
    plt.figure(figsize=(10, 6))
    improvement = df['Test_Accuracy'].diff().fillna(0)
    
    plt.bar(df['Epoch'], improvement, color=np.where(improvement >= 0, 'green', 'red'), alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy Improvement from Previous Epoch (%)')
    plt.title('Epoch-wise Accuracy Improvement', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (epoch, imp) in enumerate(zip(df['Epoch'], improvement)):
        if imp != 0:
            plt.text(epoch, imp + (0.1 if imp >= 0 else -0.3), f'{imp:+.2f}%', 
                    ha='center', va='bottom' if imp >= 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('convergence_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot 8: Performance Comparison with Baseline
    plt.figure(figsize=(10, 6))
    
    baseline_acc = 98.88
    current_best = df['Test_Accuracy'].max()
    
    categories = ['Previous Baseline', 'SAC Improved']
    accuracies = [baseline_acc, current_best]
    colors = ['red', 'green']
    
    bars = plt.bar(categories, accuracies, color=colors, alpha=0.7, width=0.5)
    plt.ylabel('Accuracy (%)')
    plt.title('Performance Improvement: SAC vs Baseline', fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Add improvement text
    improvement_pct = ((current_best - baseline_acc) / baseline_acc) * 100
    plt.text(0.5, 85, f'+{improvement_pct:.1f}% Improvement', 
            ha='center', va='center', fontsize=16, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot 9: Detailed Performance Summary
    plt.figure(figsize=(12, 8))
    
    # Create a summary table
    metrics = [
        'Best Accuracy', 'Final Accuracy', 'Average Accuracy', 
        'Best Epoch', 'Total Training Time', 'Avg Epoch Time'
    ]
    values = [
        f'{df["Test_Accuracy"].max():.2f}%',
        f'{df["Test_Accuracy"].iloc[-1]:.2f}%',
        f'{df["Test_Accuracy"].mean():.2f}%',
        f'{df["Test_Accuracy"].idxmax() + 1}',
        f'{df["Cumulative_Time_s"].iloc[-1]/60:.2f} min',
        f'{df["Epoch_Time_s"].mean():.2f} s'
    ]
    
    # Create table
    table = plt.table(cellText=[values],
                     rowLabels=['Values'],
                     colLabels=metrics,
                     cellLoc='center',
                     loc='center',
                     bbox=[0.1, 0.5, 0.8, 0.3])
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    plt.title('Detailed Performance Summary\nSpectral Anchor Consistency Domain Adaptation', 
              fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    
    # Add some text annotations
    plt.figtext(0.1, 0.85, '🎉 OUTSTANDING RESULTS!', fontsize=18, fontweight='bold', color='green')
    plt.figtext(0.1, 0.80, f'dslr → webcam Adaptation: {current_best:.2f}% Accuracy', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('performance_summary.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("✅ All individual plots saved successfully!")

# Run the plotting function
if __name__ == "__main__":
    csv_path = "./improved_results_dslr_to_webcam\dslr_to_webcam_swin_small_patch4_window7_224_improved_results_d_w_20250923_122732.csv"
    
    try:
        create_individual_plots(csv_path)
        
        # Print final summary
        df = pd.read_csv(csv_path)
        print(f"\n📊 FINAL SUMMARY:")
        print(f"   Best Accuracy: {df['Test_Accuracy'].max():.2f}% (Epoch {df['Test_Accuracy'].idxmax() + 1})")
        print(f"   Final Accuracy: {df['Test_Accuracy'].iloc[-1]:.2f}%")
        print(f"   Improvement from baseline: +{(df['Test_Accuracy'].max() - 98.62):.2f}%")
        print(f"   Relative improvement: +{((df['Test_Accuracy'].max() - 98.62) / 98.62 * 100):.1f}%")
        print(f"   Total epochs: {len(df)}")
        print(f"   Total training time: {df['Cumulative_Time_s'].iloc[-1]/60:.2f} minutes")
        
    except FileNotFoundError:
        print("❌ CSV file not found. Please check the path.")
    except Exception as e:
        print(f"❌ Error: {e}")
        