import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Setup
# =========================
csv_path = r'C:\\Users\\Admin\\Aruna_jrf_projects\\Datasets\\Office-31\\paper_experiments\\improved_results_webcam_to_amazon\\webcam_to_amazon_swin_small_patch4_window7_224_improved_results_1__w_a_20250925_135858.csv'
plot_dir = r'C:\\Users\\Admin\\Aruna_jrf_projects\\Datasets\\Office-31\\paper_experiments\\sac_DA_plots\\w_a_bigsize'
os.makedirs(plot_dir, exist_ok=True)

# Load CSV
df = pd.read_csv(csv_path)

# Baseline for comparison
baseline_acc = 65.98

# Check data ranges
print("Time data range:", df['Epoch_Time_s'].min(), "-", df['Epoch_Time_s'].max())
print("Accuracy data range:", df['Test_Accuracy'].min(), "-", df['Test_Accuracy'].max())

# =========================
# Enhanced IEEE Style Configuration
# =========================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 11,           # Increased base font size
    "axes.titlesize": 13,      # Larger titles
    "axes.labelsize": 12,      # Larger axis labels
    "xtick.labelsize": 10,     # Larger tick labels
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "axes.linewidth": 1.2,     # Thicker axes lines
    "grid.alpha": 0.5,         # More visible grid
    "grid.linewidth": 0.8,
    "lines.linewidth": 2.5,    # Thicker lines
    "lines.markersize": 7,     # Larger markers
})

# =========================
# 1. FIXED Dual-axis Plot (Single Column)
# =========================
fig, ax1 = plt.subplots(figsize=(4.2, 3.5))  # Slightly larger for better visibility

# Primary axis - Accuracy
ax1.plot(df['Epoch'], df['Test_Accuracy'], 'r-o', 
         linewidth=2.5, markersize=5, markevery=2, 
         label='Test Accuracy', zorder=3, markerfacecolor='red', markeredgecolor='red')
ax1.axhline(y=baseline_acc, color='gray', linestyle='--', 
            linewidth=2.0, label='Baseline', alpha=0.9)
ax1.set_xlabel("Epoch", fontweight="bold", fontsize=12)
ax1.set_ylabel("Accuracy (%)", color="red", fontweight="bold", fontsize=12)
ax1.tick_params(axis="y", labelcolor="red", labelsize=10)
ax1.tick_params(axis="x", labelsize=10)
ax1.set_ylim(60, 100)
ax1.set_xlim(0, len(df)+1)
ax1.grid(True, linestyle=':', alpha=0.6, linewidth=0.9)

# Secondary axis - Time (FIXED: More visible blue line)
ax2 = ax1.twinx()
ax2.plot(df['Epoch'], df['Epoch_Time_s'], 'b-s', 
         linewidth=2.5, markersize=5, markevery=2, 
         label='Time per Epoch', alpha=1.0, markerfacecolor='blue', 
         markeredgecolor='blue')
ax2.set_ylabel("Time per Epoch (s)", color="blue", fontweight="bold", fontsize=12)
ax2.tick_params(axis="y", labelcolor="blue", labelsize=10)

time_min = df['Epoch_Time_s'].min()
time_max = df['Epoch_Time_s'].max()
ax2.set_ylim(time_min - 2, time_max + 2) 
# FIXED: Better legend positioning that doesn't cover the lines
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
# Try different positions - choose the one that works best:
# Option 1: Upper left (usually empty space)
ax1.legend(lines1 + lines2, labels1 + labels2, 
           loc='upper left', frameon=True, framealpha=0.98,
           edgecolor='black', facecolor='white', fontsize=11,
           bbox_to_anchor=(0.02, 0.98))

# Option 2: Upper center (if upper left doesn't work well)
# ax1.legend(lines1 + lines2, labels1 + labels2, 
#            loc='upper center', bbox_to_anchor=(0.5, 1.15),
#            frameon=True, framealpha=0.98, ncol=3,
#            edgecolor='black', facecolor='white', fontsize=11)

# Option 3: Lower center (below the plot)
# ax1.legend(lines1 + lines2, labels1 + labels2, 
#            loc='upper center', bbox_to_anchor=(0.5, -0.15),
#            frameon=True, framealpha=0.98, ncol=3,
#            edgecolor='black', facecolor='white', fontsize=11)

plt.title("Accuracy vs Training Time (W→A)", fontweight="bold", fontsize=13, pad=15)
plt.tight_layout()
plt.savefig(f"{plot_dir}/W_A_accuracy_time_fixed.png", dpi=600, bbox_inches="tight")
plt.savefig(f"{plot_dir}/W_A_accuracy_time_fixed.pdf", dpi=600, bbox_inches="tight")
plt.close()

# =========================
# 2. Enhanced 2×2 Grid Plot (Double Column)
# =========================
fig, axs = plt.subplots(2, 2, figsize=(8.0, 6.0))  # Increased size for better visibility

# Top-left: Accuracy with enhanced visibility
axs[0,0].plot(df['Epoch'], df['Test_Accuracy'], 'r-o', 
              linewidth=2.5, markersize=7, markevery=2, 
              markerfacecolor='red', markeredgecolor='darkred')
axs[0,0].axhline(y=baseline_acc, color='gray', linestyle='--', linewidth=2.0, alpha=0.9)
axs[0,0].set_ylabel("Accuracy (%)", fontweight="bold", fontsize=12)
axs[0,0].set_ylim(60, 100)
# Enhanced subplot label
axs[0,0].text(0.03, 0.95, '(a) Test Accuracy', transform=axs[0,0].transAxes, 
              fontsize=12, fontweight='bold', va='top', 
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor='black'))
axs[0,0].grid(True, linestyle=':', alpha=0.6, linewidth=0.9)

# Top-right: Loss with enhanced visibility
axs[0,1].semilogy(df['Epoch'], df['Train_CLS_Loss'], 'g-', label="Cls Loss", linewidth=2.5)
axs[0,1].semilogy(df['Epoch'], df['Train_SAC_Loss'], 'b-', label="SAC Loss", linewidth=2.5)
axs[0,1].semilogy(df['Epoch'], df['Test_Loss'], 'r--', label="Test Loss", linewidth=2.5)
axs[0,1].set_ylabel("Loss (log scale)", fontweight="bold", fontsize=12)
axs[0,1].text(0.03, 0.95, '(b) Loss Components', transform=axs[0,1].transAxes, 
              fontsize=12, fontweight='bold', va='top',
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor='black'))
axs[0,1].legend(frameon=True, framealpha=0.95, edgecolor='black', 
                fontsize=10, loc='upper right')
axs[0,1].grid(True, linestyle=':', alpha=0.6, linewidth=0.9)

# Bottom-left: Time with enhanced bars
width = 0.7
bars = axs[1,0].bar(df['Epoch'], df['Epoch_Time_s'], width, 
                    color="lightblue", alpha=0.9, edgecolor='blue', linewidth=1.0,
                    label="Epoch Time")
axs[1,0].plot(df['Epoch'], df['Cumulative_Time_s']/60, 'k-', 
              linewidth=2.5, label="Cumulative (min)")
axs[1,0].set_xlabel("Epoch", fontweight="bold", fontsize=12)
axs[1,0].set_ylabel("Time (s/min)", fontweight="bold", fontsize=12)
axs[1,0].text(0.03, 0.95, '(c) Training Time', transform=axs[1,0].transAxes, 
              fontsize=12, fontweight='bold', va='top',
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor='black'))
axs[1,0].legend(frameon=True, framealpha=0.95, edgecolor='black', 
                fontsize=10, loc='upper left')
axs[1,0].grid(True, linestyle=':', alpha=0.6, linewidth=0.9)

# Bottom-right: Confidence with enhanced markers
axs[1,1].plot(df['Epoch'], df['Conf_Threshold'], 'm-^', 
              linewidth=2.5, markersize=7, markevery=2, 
              markerfacecolor='magenta', markeredgecolor='darkmagenta')
axs[1,1].set_xlabel("Epoch", fontweight="bold", fontsize=12)
axs[1,1].set_ylabel("Confidence Threshold", fontweight="bold", fontsize=12)
axs[1,1].set_ylim(0.75, 1.0)
axs[1,1].text(0.03, 0.95, '(d) Confidence Threshold', transform=axs[1,1].transAxes, 
              fontsize=12, fontweight='bold', va='top',
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor='black'))
axs[1,1].grid(True, linestyle=':', alpha=0.6, linewidth=0.9)

# Enhanced final adjustments
for ax in axs.flat:
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.set_xlim(0, len(df)+1)
    # Thicker spine lines
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

plt.suptitle("SAC Domain Adaptation Analysis: Webcam to Amazon (Office-31)", 
             fontsize=15, fontweight="bold", y=0.98)
plt.tight_layout()
plt.subplots_adjust(top=0.93, wspace=0.3, hspace=0.4)

plt.savefig(f"{plot_dir}/W_A_training_analysis_fixed.png", dpi=600, bbox_inches="tight")
plt.savefig(f"{plot_dir}/W_A_training_analysis_fixed.pdf", dpi=600, bbox_inches="tight")
plt.close()

# =========================
# 3. Alternative: Separate Time Plot for Better Visibility
# =========================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.0, 6.0))

# Top: Accuracy only
ax1.plot(df['Epoch'], df['Test_Accuracy'], 'r-o', linewidth=2.5, markersize=7, 
         label='Test Accuracy', markevery=2)
ax1.axhline(y=baseline_acc, color='gray', linestyle='--', linewidth=2.0, 
            label='Baseline', alpha=0.9)
ax1.set_ylabel("Accuracy (%)", fontweight="bold", fontsize=12)
ax1.set_ylim(60, 100)
ax1.legend(frameon=True, framealpha=0.95, edgecolor='black', fontsize=11)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.set_title("Test Accuracy (W→A)", fontweight="bold", fontsize=13)

# Bottom: Time only
ax2.plot(df['Epoch'], df['Epoch_Time_s'], 'b-s', linewidth=2.5, markersize=7, 
         label='Time per Epoch', markevery=2, markerfacecolor='blue')
ax2.set_xlabel("Epoch", fontweight="bold", fontsize=12)
ax2.set_ylabel("Time (s)", fontweight="bold", fontsize=12)
ax2.legend(frameon=True, framealpha=0.95, edgecolor='black', fontsize=11)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.set_title("Training Time per Epoch", fontweight="bold", fontsize=13)

plt.tight_layout()
plt.savefig(f"{plot_dir}/W_A_separate_plots.png", dpi=600, bbox_inches="tight")
plt.savefig(f"{plot_dir}/W_A_separate_plots.pdf", dpi=600, bbox_inches="tight")
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
print("SAC DOMAIN ADAPTATION SUMMARY - Webcam to Amazon")
print("="*60)
print(f"{'Best Accuracy:':<25} {best_acc:>6.2f}% (Epoch {best_epoch:>2d})")
print(f"{'Final Accuracy:':<25} {final_acc:>6.2f}%")
print(f"{'Baseline Accuracy:':<25} {baseline_acc:>6.2f}%")
print(f"{'Improvement:':<25} {best_acc - baseline_acc:>+6.2f}%")
print(f"{'Total Training Time:':<25} {total_time_min:>6.1f} min")
print(f"{'Average Epoch Time:':<25} {avg_epoch_time:>6.1f} s")
print(f"{'Time Range:':<25} {df['Epoch_Time_s'].min():>6.1f} - {df['Epoch_Time_s'].max():.1f} s")
print(f"{'Confidence Range:':<25} {df['Conf_Threshold'].min():>6.3f} - {df['Conf_Threshold'].max():.3f}")
print("="*60)