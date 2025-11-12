import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load your results
df = pd.read_csv('C:\\Users\\Admin\\Aruna_jrf_projects\\Datasets\\Office-31\\paper_experiments\\results\\swin_t_mkmmd_a_d_results_20250917_104601.csv')

# Create output directory for plots
plot_dir = './results/plots/mkmmd'
os.makedirs(plot_dir, exist_ok=True)

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'DejaVu Sans'

# 1. Accuracy Comparison Graph (Main Result)
plt.figure(figsize=(14, 8))
plt.plot(df['Epoch'], df['Test_Accuracy'], 'ro-', linewidth=2.5, markersize=8, label='Test Accuracy (Target - DSLR)')
plt.plot(df['Epoch'], df['Train_Accuracy'], 'bo-', linewidth=2.5, markersize=8, label='Train Accuracy (Source - AMAZON )', alpha=0.7)

plt.xlabel('Epoch', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
plt.title('MKMMD Performance: A → D Domain Adaptation\n(Swin-T Backbone, 50 Epochs)', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(np.arange(0, 51, 5))

# Find and annotate the best point
best_epoch = df['Test_Accuracy'].idxmax() + 1
best_acc = df['Test_Accuracy'].max()

plt.annotate(f'Best: {best_acc:.1f}% (Epoch {best_epoch})', 
             xy=(best_epoch, best_acc), xytext=(best_epoch+3, best_acc+3),
             arrowprops=dict(facecolor='green', shrink=0.05, width=2, headwidth=8),
             fontsize=12, color='green', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{plot_dir}/1. mkmmd_a_d_accuracy_curve.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{plot_dir}/1. mkmmd_a_d_accuracy_curve.pdf', bbox_inches='tight')
plt.show()

# 2. Loss Components Analysis
plt.figure(figsize=(14, 8))
plt.plot(df['Epoch'], df['Train_Loss'], 'g-', linewidth=2.5, label='Classification Loss (Source)')
plt.plot(df['Epoch'], df['MMD_Loss'], 'b-', linewidth=2.5, label='MKMMD Loss (Target)')
plt.plot(df['Epoch'], df['Test_Loss'], 'r-', linewidth=2.5, label='Test Loss (Target)', alpha=0.7)

plt.xlabel('Epoch', fontsize=14, fontweight='bold')
plt.ylabel('Loss Value', fontsize=14, fontweight='bold')
plt.title('Loss Components: MKMMD Training Dynamics\n(A → D Domain Adaptation)', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.yscale('log')  # Log scale to see all components clearly
plt.xticks(np.arange(0, 51, 5))

plt.tight_layout()
plt.savefig(f'{plot_dir}/2. mkmmd_a_d_loss_components.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{plot_dir}/2. mkmmd_a_d_loss_components.pdf', bbox_inches='tight')
plt.show()

# 3. Timing Analysis
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
plt.title('Computational Efficiency: Training Time per Epoch\n(Swin-T + MKMMD)', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(np.arange(0, 51, 5))

plt.tight_layout()
plt.savefig(f'{plot_dir}/3. mkmmd_a_d_timing_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{plot_dir}/3. mkmmd_a_d_timing_analysis.pdf', bbox_inches='tight')
plt.show()

# 4. Cumulative Time vs Accuracy
fig, ax1 = plt.subplots(figsize=(14, 8))

color = 'tab:red'
ax1.set_xlabel('Cumulative Training Time (minutes)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Test Accuracy (%)', color=color, fontsize=14, fontweight='bold')
ax1.plot(df['Cumulative_Time_s']/60, df['Test_Accuracy'], color=color, 
         linewidth=2.5, marker='o', markersize=6, label='Test Accuracy')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Training Time per Epoch (seconds)', color=color, fontsize=14, fontweight='bold')
ax2.plot(df['Cumulative_Time_s']/60, df['Epoch_Time_s'], color=color, 
         linewidth=2.5, linestyle='--', alpha=0.7, label='Time/Epoch')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Time-Accuracy Tradeoff: MKMMD Domain Adaptation\n(A → D, Swin-T Backbone)', fontsize=16, fontweight='bold')
fig.tight_layout()
plt.savefig(f'{plot_dir}/4.mkmmd_a_d_time_accuracy_tradeoff.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{plot_dir}/4. mkmmd_a_d_time_accuracy_tradeoff.pdf', bbox_inches='tight')
plt.show()

# 5. Combined Analysis (Publication Quality)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

# Top: Accuracy
ax1.plot(df['Epoch'], df['Test_Accuracy'], 'ro-', linewidth=2.5, markersize=6, label='Test Accuracy (A)')
ax1.plot(df['Epoch'], df['Train_Accuracy'], 'bo-', linewidth=2.5, markersize=6, label='Train Accuracy (D)', alpha=0.7)
ax1.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax1.set_title('A) MKMMD Performance: A → D Domain Adaptation', fontsize=14, fontweight='bold')
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(np.arange(0, 51, 5))

# Bottom: Loss (log scale)
ax2.plot(df['Epoch'], df['Train_Loss'], 'g-', linewidth=2.5, label='Classification Loss')
ax2.plot(df['Epoch'], df['MMD_Loss'], 'b-', linewidth=2.5, label='MKMMD Loss')
ax2.plot(df['Epoch'], df['Test_Loss'], 'r-', linewidth=2.5, label='Test Loss', alpha=0.7)
ax2.set_xlabel('Epoch', fontsize=14, fontweight='bold')
ax2.set_ylabel('Loss Value (log scale)', fontsize=14, fontweight='bold')
ax2.set_title('B) Loss Components: Training Dynamics', fontsize=14, fontweight='bold')
ax2.set_yscale('log')
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(np.arange(0, 51, 5))

plt.tight_layout()
plt.savefig(f'{plot_dir}/5.mkmmd_a_d_combined_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{plot_dir}/5. mkmmd_a_d_combined_analysis.pdf', bbox_inches='tight')
plt.show()

# 6. Quantitative Analysis Report
print("=== MKMMD a_d_TRAINING ANALYSIS REPORT ===")
print(f"Best Test Accuracy: {best_acc:.2f}% (Epoch {best_epoch})")
print(f"Final Test Accuracy: {df['Test_Accuracy'].iloc[-1]:.2f}%")
print(f"Final Train Accuracy: {df['Train_Accuracy'].iloc[-1]:.2f}%")
print(f"\nLoss Reduction:")
print(f"  - Classification Loss: {df['Train_Loss'].iloc[0]:.4f} → {df['Train_Loss'].iloc[-1]:.4f} ({df['Train_Loss'].iloc[-1]/df['Train_Loss'].iloc[0]*100:.1f}%)")
print(f"  - MKMMD Loss: {df['MMD_Loss'].iloc[0]:.4f} → {df['MMD_Loss'].iloc[-1]:.4f} ({df['MMD_Loss'].iloc[-1]/df['MMD_Loss'].iloc[0]*100:.1f}%)")
print(f"\nTiming Analysis:")
print(f"  Total Training Time: {df['Cumulative_Time_s'].iloc[-1]/60:.2f} minutes")
print(f"  Average Time per Epoch: {df['Epoch_Time_s'].mean():.2f} seconds")
print(f"  Time Range: {df['Epoch_Time_s'].min():.2f} - {df['Epoch_Time_s'].max():.2f} seconds")

# 7. Save analysis summary
summary = {
    'Best_Accuracy': best_acc,
    'Best_Epoch': best_epoch,
    'Final_Test_Accuracy': df['Test_Accuracy'].iloc[-1],
    'Final_Train_Accuracy': df['Train_Accuracy'].iloc[-1],
    'CLS_Loss_Start': df['Train_Loss'].iloc[0],
    'CLS_Loss_End': df['Train_Loss'].iloc[-1],
    'MKMMD_Loss_Start': df['MMD_Loss'].iloc[0],
    'MKMMD_Loss_End': df['MMD_Loss'].iloc[-1],
    'Total_Training_Minutes': df['Cumulative_Time_s'].iloc[-1]/60,
    'Avg_Epoch_Time': df['Epoch_Time_s'].mean(),
    'Min_Epoch_Time': df['Epoch_Time_s'].min(),
    'Max_Epoch_Time': df['Epoch_Time_s'].max()
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv(f'{plot_dir}/MKMMD_a_d_performance_summary.csv', index=False)
print(f"\nAnalysis summary saved to: {plot_dir}/MKMMD_a_d_performance_summary.csv")