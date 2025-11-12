import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load your SAC results
csv_path = r'C:\\Users\\Admin\\Aruna_jrf_projects\\Datasets\\Office-31\\paper_experiments\\results_amazon_to_webcam\\amazon_to_webcam_swin_tiny_patch4_window7_224_sac_single_a_w_results_20250916_160015.csv'
df = pd.read_csv(csv_path)

# Create output directory for plots
plot_dir = './results/sac_single_plots'
os.makedirs(plot_dir, exist_ok=True)

print("✓ Loaded SAC training data!")
print(f"Best Accuracy: {df['Test_Accuracy'].max():.2f}%")

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'DejaVu Sans'

# 1. MAIN RESULT: Accuracy Progression
plt.figure(figsize=(14, 8))
plt.plot(df['Epoch'], df['Test_Accuracy'], 'ro-', linewidth=2.5, markersize=8, label='Test Accuracy (Target Domain)')

# Find and annotate the best point
best_epoch = df['Test_Accuracy'].idxmax() + 1
best_acc = df['Test_Accuracy'].max()

plt.axhline(y=78.06, color='red', linestyle='--', linewidth=2, label='Baseline (78.06)')
plt.axhline(y=best_acc, color='green', linestyle='--', linewidth=2, label=f'Best SAC ({best_acc:.1f}%)')

plt.xlabel('Epoch', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
plt.title('SAC Performance: Domain Adaptation with Spectral Consistency\n(Swin-T Backbone)', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim(0, 100)

# Annotate best point
plt.annotate(f'Best: {best_acc:.1f}% (Epoch {best_epoch})', 
             xy=(best_epoch, best_acc), xytext=(best_epoch+3, best_acc+3),
             arrowprops=dict(facecolor='green', shrink=0.05, width=2, headwidth=8),
             fontsize=12, color='green', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{plot_dir}/1. sac_a_w_accuracy_curve.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{plot_dir}/1. sac_a_w_accuracy_curve.pdf', bbox_inches='tight')
plt.show()

# 2. LOSS COMPONENTS ANALYSIS
plt.figure(figsize=(14, 8))
plt.plot(df['Epoch'], df['Train_CLS_Loss'], 'g-', linewidth=2.5, label='Classification Loss')
plt.plot(df['Epoch'], df['Train_SAC_Loss'], 'b-', linewidth=2.5, label='SAC Consistency Loss')
plt.plot(df['Epoch'], df['Test_Loss'], 'r-', linewidth=2.5, label='Test Loss', alpha=0.7)

plt.xlabel('Epoch', fontsize=14, fontweight='bold')
plt.ylabel('Loss Value', fontsize=14, fontweight='bold')
plt.title('Loss Components: SAC Training Dynamics\n(Spectral Adversarial Consistency)', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.yscale('log')  # Log scale to see all components clearly

plt.tight_layout()
plt.savefig(f'{plot_dir}/2. sac_a_w_loss_components.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{plot_dir}/2. sac_a_w_loss_components.pdf', bbox_inches='tight')
plt.show()

# 3. CONFIDENCE THRESHOLD ANALYSIS
plt.figure(figsize=(14, 8))

plt.plot(df['Epoch'], df['Conf_Threshold'], 'purple', linewidth=2.5, marker='s', markersize=6, label='Confidence Threshold')
plt.xlabel('Epoch', fontsize=14, fontweight='bold')
plt.ylabel('Confidence Threshold', fontsize=14, fontweight='bold')
plt.title('Confidence Threshold Schedule: Reverse Curriculum Learning', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.ylim(0.75, 1.0)

plt.tight_layout()
plt.savefig(f'{plot_dir}/3. sac_a_w_confidence_schedule.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{plot_dir}/3. sac_a_w_confidence_schedule.pdf', bbox_inches='tight')
plt.show()

# 4. TIMING ANALYSIS
plt.figure(figsize=(12, 6))
plt.bar(df['Epoch'], df['Epoch_Time_s'], color='skyblue', alpha=0.7, edgecolor='navy')

# Calculate statistics
avg_time = df['Epoch_Time_s'].mean()
min_time = df['Epoch_Time_s'].min()
max_time = df['Epoch_Time_s'].max()

plt.axhline(y=avg_time, color='red', linestyle='--', linewidth=2, 
            label=f'Average: {avg_time:.1f}s/epoch')
plt.axhline(y=min_time, color='green', linestyle=':', linewidth=2, 
            label=f'Minimum: {min_time:.1f}s')
plt.axhline(y=max_time, color='orange', linestyle=':', linewidth=2, 
            label=f'Maximum: {max_time:.1f}s')

plt.xlabel('Epoch', fontsize=14, fontweight='bold')
plt.ylabel('Time per Epoch (seconds)', fontsize=14, fontweight='bold')
plt.title('Computational Efficiency: SAC Training Time\n(Swin-T + Spectral Consistency)', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{plot_dir}/4. sac_a_d_timing_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{plot_dir}/4. sac_a_d_timing_analysis.pdf', bbox_inches='tight')
plt.show()

# 5. COMBINED ANALYSIS (Publication Quality)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

# Top: Accuracy
ax1.plot(df['Epoch'], df['Test_Accuracy'], 'ro-', linewidth=2.5, markersize=6, label='Test Accuracy')
ax1.axhline(y=78.06, color='red', linestyle='--', linewidth=2, label='Baseline (78.06%)')
ax1.axhline(y=best_acc, color='green', linestyle='--', linewidth=2, label=f'Best SAC ({best_acc:.1f}%)')
ax1.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax1.set_title('A) SAC Performance: Domain Adaptation Results', fontsize=14, fontweight='bold')
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 100)

# Bottom: Confidence Threshold
ax2.plot(df['Epoch'], df['Conf_Threshold'], 'purple', linewidth=2.5, marker='s', markersize=6)
ax2.set_xlabel('Epoch', fontsize=14, fontweight='bold')
ax2.set_ylabel('Confidence Threshold', fontsize=14, fontweight='bold')
ax2.set_title('B) Confidence Schedule: Reverse Curriculum', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0.75, 1.0)

plt.tight_layout()
plt.savefig(f'{plot_dir}/5. sac_a_w_combined_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{plot_dir}/5. sac_a_w_combined_analysis.pdf', bbox_inches='tight')
plt.show()

# 6. PERFORMANCE COMPARISON
plt.figure(figsize=(10, 6))

methods = ['Source-Only Baseline', 'SAC Method (Yours)']
accuracies = [78.06, best_acc]
colors = ['red', 'green']

bars = plt.bar(methods, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
plt.title('Performance Comparison: SAC vs Baseline\n(office Domain Adaptation)', fontsize=16, fontweight='bold')
plt.ylim(0, 100)

# Add value labels on bars
for bar, accuracy in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{accuracy:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{plot_dir}/6. sac_a_w_vs_baseline.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{plot_dir}/6. sac_a_w_vs_baseline.pdf', bbox_inches='tight')
plt.show()

# 7. QUANTITATIVE ANALYSIS REPORT
print("=== SAC TRAINING ANALYSIS REPORT ===")
print(f"Best Test Accuracy: {best_acc:.2f}% (Epoch {best_epoch})")
print(f"Final Test Accuracy: {df['Test_Accuracy'].iloc[-1]:.2f}%")
print(f"Baseline Accuracy: 78.06%")
print(f"Improvement: {best_acc-78.06:+.2f}%")

print(f"\nLoss Analysis:")
print(f"  - Final Classification Loss: {df['Train_CLS_Loss'].iloc[-1]:.4f}")
print(f"  - Final SAC Consistency Loss: {df['Train_SAC_Loss'].iloc[-1]:.4f}")
print(f"  - Final Test Loss: {df['Test_Loss'].iloc[-1]:.4f}")

print(f"\nTiming Analysis:")
print(f"  Total Training Time: {df['Cumulative_Time_s'].iloc[-1]/60:.2f} minutes")
print(f"  Average Time per Epoch: {df['Epoch_Time_s'].mean():.2f} seconds")
print(f"  Time Range: {df['Epoch_Time_s'].min():.2f} - {df['Epoch_Time_s'].max():.2f} seconds")

print(f"\nConfidence Threshold Range: {df['Conf_Threshold'].min():.3f} - {df['Conf_Threshold'].max():.3f}")

# 8. SAVE ANALYSIS SUMMARY
summary = {
    'Best_Accuracy': best_acc,
    'Best_Epoch': best_epoch,
    'Final_Test_Accuracy': df['Test_Accuracy'].iloc[-1],
    'Baseline_Accuracy': 78.06,
    'Improvement': best_acc - 78.06,
    'Final_CLS_Loss': df['Train_CLS_Loss'].iloc[-1],
    'Final_SAC_Loss': df['Train_SAC_Loss'].iloc[-1],
    'Final_Test_Loss': df['Test_Loss'].iloc[-1],
    'Total_Training_Minutes': df['Cumulative_Time_s'].iloc[-1]/60,
    'Avg_Epoch_Time': df['Epoch_Time_s'].mean(),
    'Min_Confidence': df['Conf_Threshold'].min(),
    'Max_Confidence': df['Conf_Threshold'].max()
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv(f'{plot_dir}/sac_a_w_performance_summary.csv', index=False)
print(f"\nAnalysis summary saved to: {plot_dir}/sac__a_w_performance_summary.csv")

print(f"\n🎉 All plots saved to: {plot_dir}/")
print("✓ 1. sac__a_w_accuracy_curve.png")
print("✓ 2. sac__a_w_loss_components.png") 
print("✓ 3. sac__a_w_confidence_schedule.png")
print("✓ 4. sac__a_w_timing_analysis.png")
print("✓ 5. sac__a_w_combined_analysis.png")
print("✓ 6. sac__a_w_vs_baseline.png")
print("✓ sac__a_w_performance_summary.csv")