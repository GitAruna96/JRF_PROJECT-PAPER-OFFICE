import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Setup
# =========================
csv_path = r'C:\\Users\\Admin\\Aruna_jrf_projects\\Datasets\\Office-31\\paper_experiments\\improved_results_dslr_to_amazon\\dslr_to_amazon_swin_small_patch4_window7_224_improved_results1__d_a_20250925_125530.csv'
plot_dir = r'C:\\Users\\Admin\\Aruna_jrf_projects\\Datasets\\Office-31\\paper_experiments\\sac_DA_plots\\d_a_bigsize'
os.makedirs(plot_dir, exist_ok=True)

# Load CSV
df = pd.read_csv(csv_path)

# Baseline for comparison
baseline_acc = 65.20

# =========================
# Enhanced IEEE Style Configuration
# =========================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 10,           # Increased base font size
    "axes.titlesize": 12,      # Larger titles
    "axes.labelsize": 11,      # Larger axis labels
    "xtick.labelsize": 9,      # Larger tick labels
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "axes.linewidth": 1.0,     # Thicker axes lines
    "grid.alpha": 0.4,         # Slightly more visible grid
    "grid.linewidth": 0.6,
    "lines.linewidth": 2.0,    # Thicker lines
    "lines.markersize": 5,     # Larger markers
})

# =========================
# 1. Enhanced Dual-axis Plot (Single Column)
# =========================
fig, ax1 = plt.subplots(figsize=(4.0, 3.2))  # Increased size for better visibility

# Primary axis - Accuracy
ax1.plot(df['Epoch'], df['Test_Accuracy'], 'r-o', 
         linewidth=2.0, markersize=5, markevery=2, 
         label='Test Accuracy', zorder=3, markerfacecolor='red')
ax1.axhline(y=baseline_acc, color='gray', linestyle='--', 
            linewidth=1.5, label='Baseline', alpha=0.9)
ax1.set_xlabel("Epoch", fontweight="bold", fontsize=11)
ax1.set_ylabel("Accuracy (%)", color="red", fontweight="bold", fontsize=11)
ax1.tick_params(axis="y", labelcolor="red", labelsize=9)
ax1.tick_params(axis="x", labelsize=9)
ax1.set_ylim(60, 100)
ax1.set_xlim(0, len(df)+1)
ax1.grid(True, linestyle=':', alpha=0.5, linewidth=0.8)

# Secondary axis - Time
ax2 = ax1.twinx()
ax2.plot(df['Epoch'], df['Epoch_Time_s'], 'b-s', 
         linewidth=2.0, markersize=5, markevery=2, 
         label='Time per Epoch', alpha=0.9, markerfacecolor='blue')
ax2.set_ylabel("Time per Epoch (s)", color="blue", fontweight="bold", fontsize=11)
ax2.tick_params(axis="y", labelcolor="blue", labelsize=9)
ax2.set_ylim(40, 60)

# Enhanced legend with better visibility
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, 
           loc='lower right', frameon=True, framealpha=0.95,
           edgecolor='black', facecolor='white', fontsize=10)

plt.title("Accuracy vs Training Time (D→A)", fontweight="bold", fontsize=12, pad=12)
plt.tight_layout()
plt.savefig(f"{plot_dir}/D_A_accuracy_time_enhanced.png", dpi=600, bbox_inches="tight")
plt.savefig(f"{plot_dir}/D_A_accuracy_time_enhanced.pdf", dpi=600, bbox_inches="tight")
plt.close()

# =========================
# 2. Enhanced 2×2 Grid Plot (Double Column)
# =========================
fig, axs = plt.subplots(2, 2, figsize=(7.5, 5.5))  # Increased size for better visibility

# Top-left: Accuracy with enhanced visibility
axs[0,0].plot(df['Epoch'], df['Test_Accuracy'], 'r-o', 
              linewidth=2.0, markersize=5, markevery=2, markerfacecolor='red')
axs[0,0].axhline(y=baseline_acc, color='gray', linestyle='--', linewidth=1.5, alpha=0.9)
axs[0,0].set_ylabel("Accuracy (%)", fontweight="bold", fontsize=11)
axs[0,0].set_ylim(60, 100)
# Enhanced subplot label
axs[0,0].text(0.03, 0.95, '(a) Test Accuracy', transform=axs[0,0].transAxes, 
              fontsize=11, fontweight='bold', va='top', 
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor='gray'))
axs[0,0].grid(True, linestyle=':', alpha=0.5, linewidth=0.8)

# Top-right: Loss with enhanced visibility
axs[0,1].semilogy(df['Epoch'], df['Train_CLS_Loss'], 'g-', label="Cls Loss", linewidth=2.0)
axs[0,1].semilogy(df['Epoch'], df['Train_SAC_Loss'], 'b-', label="SAC Loss", linewidth=2.0)
axs[0,1].semilogy(df['Epoch'], df['Test_Loss'], 'r--', label="Test Loss", linewidth=2.0)
axs[0,1].set_ylabel("Loss (log scale)", fontweight="bold", fontsize=11)
axs[0,1].text(0.03, 0.95, '(b) Loss Components', transform=axs[0,1].transAxes, 
              fontsize=11, fontweight='bold', va='top',
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor='gray'))
axs[0,1].legend(frameon=True, framealpha=0.9, edgecolor='black', fontsize=9)
axs[0,1].grid(True, linestyle=':', alpha=0.5, linewidth=0.8)

# Bottom-left: Time with enhanced bars
width = 0.6
bars = axs[1,0].bar(df['Epoch'], df['Epoch_Time_s'], width, 
                    color="lightblue", alpha=0.8, edgecolor='blue', linewidth=0.5,
                    label="Epoch Time")
axs[1,0].plot(df['Epoch'], df['Cumulative_Time_s']/60, 'k-', 
              linewidth=2.0, label="Cumulative (min)")
axs[1,0].set_xlabel("Epoch", fontweight="bold", fontsize=11)
axs[1,0].set_ylabel("Time (s/min)", fontweight="bold", fontsize=11)
axs[1,0].text(0.03, 0.95, '(c) Training Time', transform=axs[1,0].transAxes, 
              fontsize=11, fontweight='bold', va='top',
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor='gray'))
axs[1,0].legend(frameon=True, framealpha=0.9, edgecolor='black', fontsize=9)
axs[1,0].grid(True, linestyle=':', alpha=0.5, linewidth=0.8)

# Bottom-right: Confidence with enhanced markers
axs[1,1].plot(df['Epoch'], df['Conf_Threshold'], 'm-^', 
              linewidth=2.0, markersize=6, markevery=2, markerfacecolor='magenta')
axs[1,1].set_xlabel("Epoch", fontweight="bold", fontsize=11)
axs[1,1].set_ylabel("Confidence Threshold", fontweight="bold", fontsize=11)
axs[1,1].set_ylim(0.75, 1.0)
axs[1,1].text(0.03, 0.95, '(d) Confidence Threshold', transform=axs[1,1].transAxes, 
              fontsize=11, fontweight='bold', va='top',
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor='gray'))
axs[1,1].grid(True, linestyle=':', alpha=0.5, linewidth=0.8)

# Enhanced final adjustments
for ax in axs.flat:
    ax.tick_params(axis='both', which='major', labelsize=10)  # Larger tick labels
    ax.set_xlim(0, len(df)+1)
    # Thicker spine lines
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

plt.suptitle("SAC Domain Adaptation Analysis: DSLR to Amazon (Office-31)", 
             fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout()
plt.subplots_adjust(top=0.93, wspace=0.35, hspace=0.45)  # More spacing

plt.savefig(f"{plot_dir}/D_A_training_analysis_enhanced.png", dpi=600, bbox_inches="tight")
plt.savefig(f"{plot_dir}/D_A_training_analysis_enhanced.pdf", dpi=600, bbox_inches="tight")
plt.close()

# =========================
# 3. Additional: Single Column Multi-line Plot (Alternative)
# =========================
fig, ax = plt.subplots(figsize=(4.0, 3.5))

# Plot multiple metrics with enhanced visibility
ax.plot(df['Epoch'], df['Test_Accuracy'], 'r-o', linewidth=2.5, markersize=6, 
        label='Test Accuracy', markevery=2)
ax.plot(df['Epoch'], df['Conf_Threshold'] * 100, 'g--s', linewidth=2.0, markersize=5,
        label='Confidence × 100', markevery=2)
ax.axhline(y=baseline_acc, color='gray', linestyle='--', linewidth=2.0, 
           label='Baseline', alpha=0.8)

ax.set_xlabel("Epoch", fontweight="bold", fontsize=12)
ax.set_ylabel("Accuracy / Confidence (%)", fontweight="bold", fontsize=12)
ax.set_ylim(70, 105)
ax.legend(frameon=True, framealpha=0.95, edgecolor='black', fontsize=10,
          loc='lower right')
ax.grid(True, linestyle=':', alpha=0.6, linewidth=0.8)
ax.tick_params(axis='both', which='major', labelsize=10)

plt.title("Accuracy and Confidence Evolution (D→A)", fontweight="bold", fontsize=13, pad=15)
plt.tight_layout()
plt.savefig(f"{plot_dir}/D_A_accuracy_enhanced_confidence_combined.png", dpi=600, bbox_inches="tight")
plt.savefig(f"{plot_dir}/D_A_accuracy_enhanced_confidence_combined.pdf", dpi=600, bbox_inches="tight")
plt.close()

# =========================
# 4. Enhanced Summary Statistics
# =========================
best_epoch = df['Test_Accuracy'].idxmax() + 1
best_acc = df['Test_Accuracy'].max()
final_acc = df['Test_Accuracy'].iloc[-1]
total_time_min = df['Cumulative_Time_s'].iloc[-1] / 60
avg_epoch_time = df['Epoch_Time_s'].mean()

print("\n" + "="*60)
print("SAC DOMAIN ADAPTATION SUMMARY - DSLR to Amazon")
print("="*60)
print(f"{'Best Accuracy:':<25} {best_acc:>6.2f}% (Epoch {best_epoch:>2d})")
print(f"{'Final Accuracy:':<25} {final_acc:>6.2f}%")
print(f"{'Baseline Accuracy:':<25} {baseline_acc:>6}%")
print(f"{'Improvement:':<25} {best_acc - baseline_acc:>+6.2f}%")
print(f"{'Total Training Time:':<25} {total_time_min:>6.1f} min")
print(f"{'Average Epoch Time:':<25} {avg_epoch_time:>6.1f} s")
print(f"{'Confidence Range:':<25} {df['Conf_Threshold'].min():>6.3f} - {df['Conf_Threshold'].max():.3f}")
print("="*60)