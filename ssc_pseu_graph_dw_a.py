import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load your results
df = pd.read_csv('C:\\Users\\Admin\\Aruna_jrf_projects\\Datasets\\Office-31\\paper_experiments\\results\\swin_t_office_ssc_pseudo__dw_a_results.csv')

# Create output directory for plots
plot_dir = './results/plots/ssc_pseu'
os.makedirs(plot_dir, exist_ok=True)

def save_figure(fig, name):
    fig.savefig(f'{plot_dir}/{name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

# 1. Accuracy Comparison Graph (The Most Important)
fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(df['Epoch'], df['Test_Accuracy'], 'ro-', linewidth=2, markersize=6, label='Test Accuracy (Target)')
ax1.plot(df['Epoch'], df['Train_Accuracy'], 'bo-', linewidth=2, markersize=6, label='Train Accuracy (Source)')
ax1.axvline(x=5.5, color='red', linestyle='--', alpha=0.7, label='Pseudo-Labeling Activated')

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Catastrophic Collapse: SSC + Pseudo-Labeling Performance\n(DW to A) Domain Adaptation)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks(df['Epoch'])

# Find and annotate the best and worst points
best_epoch = df['Test_Accuracy'].idxmax()
best_acc = df['Test_Accuracy'].max()
worst_epoch = df['Test_Accuracy'].idxmin()
worst_acc = df['Test_Accuracy'].min()

# Annotate using ax1 (not plt)
ax1.annotate(f'Best: {best_acc:.1f}% (Epoch {best_epoch+1})', 
             xy=(best_epoch+1, best_acc), xytext=(best_epoch+3, best_acc+5),
             arrowprops=dict(facecolor='green', shrink=0.05, width=2),
             fontsize=10, color='green')

ax1.annotate(f'Collapse: {worst_acc:.1f}% (Epoch {worst_epoch+1})', 
             xy=(worst_epoch+1, worst_acc), xytext=(worst_epoch-2, worst_acc-10),
             arrowprops=dict(facecolor='red', shrink=0.05, width=2),
             fontsize=10, color='red')

plt.tight_layout()
save_figure(fig1, '1.ssc_pseudo_collapse DW to A')

# 2. Loss Components Graph
fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(df['Epoch'], df['Train_Loss'], 'g-', linewidth=2, label='Classification Loss')
ax2.plot(df['Epoch'], df['SSC_Loss'], 'b-', linewidth=2, label='SSC Loss')
ax2.plot(df['Epoch'], df['Pseudo_Loss'], 'r-', linewidth=2, label='Pseudo-Labeling Loss')
ax2.axvline(x=5.5, color='red', linestyle='--', alpha=0.7, label='Pseudo-Labeling Activated')

ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss Value')
ax2.set_title('Loss Components: SSC + Pseudo-Labeling Training Dynamics')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')  # Use log scale to see all losses clearly
ax2.set_xticks(df['Epoch'])

plt.tight_layout()
save_figure(fig2, '2.DW to A loss_components')

# 3. Timing Analysis Graph
fig3, ax3 = plt.subplots(figsize=(10, 5))
# The bars are colored based on epoch number
bar_colors = ['blue'] * 5 + ['red'] * (len(df) - 5)
ax3.bar(df['Epoch'], df['Epoch_Time_s'], color=bar_colors)

# Calculate average times
avg_ssc_time = df['Epoch_Time_s'].iloc[:5].mean()
avg_pseudo_time = df['Epoch_Time_s'].iloc[5:].mean()
time_increase = ((avg_pseudo_time - avg_ssc_time) / avg_ssc_time) * 100

ax3.axhline(y=avg_ssc_time, color='green', linestyle='--', label=f'SSC-only avg: {avg_ssc_time:.1f}s')
ax3.axhline(y=avg_pseudo_time, color='orange', linestyle='--', label=f'SSC+Pseudo avg: {avg_pseudo_time:.1f}s (+{time_increase:.1f}%)')

ax3.set_xlabel('Epoch')
ax3.set_ylabel('Time per Epoch (seconds)')
ax3.set_title(f'Computational Overhead of Pseudo-Labeling\n({time_increase:.1f}% increase in epoch time)')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xticks(df['Epoch'])
plt.tight_layout()
save_figure(fig3, '3.DW to A timing_overhead')

# 4. Combined Analysis Graph (For Paper)
fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(12, 10))

# Top: Accuracy (Plot on ax4a, NOT ax1)
ax4a.plot(df['Epoch'], df['Test_Accuracy'], 'ro-', linewidth=2, markersize=6, label='Test Accuracy')
ax4a.plot(df['Epoch'], df['Train_Accuracy'], 'bo-', linewidth=2, markersize=6, label='Train Accuracy')
ax4a.axvline(x=5.5, color='red', linestyle='--', alpha=0.7)
ax4a.set_ylabel('Accuracy (%)')
ax4a.set_title('SSC + Pseudo-Labeling: Performance Collapse on DW to A Task')
ax4a.legend()
ax4a.grid(True, alpha=0.3)
ax4a.text(5.5, df['Test_Accuracy'].max()*0.8, 'Pseudo-Labeling\nActivated', 
         rotation=90, va='center', ha='right', color='red')

# Bottom: Loss (Plot on ax4b, NOT ax2)
ax4b.plot(df['Epoch'], df['Train_Loss'], 'g-', linewidth=2, label='CLS Loss')
ax4b.plot(df['Epoch'], df['SSC_Loss'], 'b-', linewidth=2, label='SSC Loss')
ax4b.plot(df['Epoch'], df['Pseudo_Loss'], 'r-', linewidth=2, label='Pseudo Loss')
ax4b.axvline(x=5.5, color='red', linestyle='--', alpha=0.7)
ax4b.set_xlabel('Epoch')
ax4b.set_ylabel('Loss Value (log scale)')
ax4b.set_yscale('log')
ax4b.legend()
ax4b.grid(True, alpha=0.3)

plt.tight_layout()
save_figure(fig4, '4.DW to A combined_analysis')

print("All figures saved successfully! Check the folder:", plot_dir)

# 5. Quantitative Analysis
print("=== QUANTITATIVE ANALYSIS ===")
print(f"Best Test Accuracy: {best_acc:.2f}% (Epoch {best_epoch+1})")
print(f"Worst Test Accuracy: {worst_acc:.2f}% (Epoch {worst_epoch+1})")
print(f"Performance Drop: {best_acc - worst_acc:.2f}% ({((best_acc - worst_acc)/best_acc)*100:.1f}% relative)")
print(f"\nTiming Analysis:")
print(f"SSC-only average time: {avg_ssc_time:.2f}s per epoch")
print(f"SSC+Pseudo average time: {avg_pseudo_time:.2f}s per epoch")
print(f"Time increase: +{time_increase:.1f}%")
print(f"\nTotal Training Time: {df['Cumulative_Time_s'].iloc[-1]/60:.2f} minutes")

# 6. Save analysis summary
summary = {
    'Best_Accuracy': best_acc,
    'Best_Epoch': best_epoch + 1, # Save as human-readable epoch number
    'Worst_Accuracy': worst_acc,
    'Worst_Epoch': worst_epoch + 1, # Save as human-readable epoch number
    'Performance_Drop_Percent': best_acc - worst_acc,
    'Performance_Drop_Relative': ((best_acc - worst_acc)/best_acc)*100,
    'Avg_SSC_Time': avg_ssc_time,
    'Avg_Pseudo_Time': avg_pseudo_time,
    'Time_Increase_Percent': time_increase,
    'Total_Training_Minutes': df['Cumulative_Time_s'].iloc[-1]/60
}

summary_df = pd.DataFrame([summary])
summary_path = f'{plot_dir}/ssc_pseudo_performance_DW to A_summary.csv'
summary_df.to_csv(summary_path, index=False)
print(f"\nAnalysis summary saved to: {summary_path}")