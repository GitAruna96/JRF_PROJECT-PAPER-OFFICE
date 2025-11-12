import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Setup
# =========================
csv_path = r'C:\\Users\\Admin\\Aruna_jrf_projects\\Datasets\\Office-31\\paper_experiments\\improved_results_amazon_to_dslr\\amazon_to_dslr_swin_small_patch4_window7_224_improved_results_20250923_101958.csv'
plot_dir = r'C:\\Users\\Admin\\Aruna_jrf_projects\\Datasets\\Office-31\\paper_experiments\\improved_plots\\a_d_acc_time'
os.makedirs(plot_dir, exist_ok=True)

# Load CSV
df = pd.read_csv(csv_path)

# Baseline for comparison
baseline_acc = 82.10

# Matplotlib style
plt.rcParams.update({
    "font.size": 12,
    "font.family": "Times New Roman",
    "axes.grid": True,
    "grid.alpha": 0.3
})

# =========================
# 1. Dual-axis Accuracy vs Time Plot
# =========================
fig, ax1 = plt.subplots(figsize=(8,5))

# Accuracy curve
ax1.plot(df['Epoch'], df['Test_Accuracy'], 'ro-', linewidth=2, markersize=6, label='Test Accuracy')
ax1.axhline(y=baseline_acc, color='gray', linestyle='--', label='Baseline')
ax1.set_xlabel("Epoch", fontsize=12, fontweight="bold")
ax1.set_ylabel("Accuracy (%)", color="red", fontsize=12, fontweight="bold")
ax1.tick_params(axis="y", labelcolor="red")
ax1.set_ylim(0, 100)

# Time curve (right axis)
ax2 = ax1.twinx()
ax2.plot(df['Epoch'], df['Epoch_Time_s'], 'b--s', linewidth=2, markersize=6, label='Time per Epoch')
ax2.set_ylabel("Time per Epoch (s)", color="blue", fontsize=12, fontweight="bold")
ax2.tick_params(axis="y", labelcolor="blue")

# Legends
fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3, fontsize=11)
plt.title("Accuracy vs Training Time (SAC)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{plot_dir}/1.accuracy_vs_time.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{plot_dir}/1.accuracy_vs_time.pdf", bbox_inches="tight")
plt.close()

# =========================
# 2. 2×2 Grid Plot: Accuracy, Loss, Time, Confidence
# =========================
fig, axs = plt.subplots(2, 2, figsize=(14,10))

# Top-left: Accuracy
axs[0,0].plot(df['Epoch'], df['Test_Accuracy'], 'r-o', linewidth=2, markersize=6, label='Test Accuracy')
axs[0,0].axhline(y=baseline_acc, color='gray', linestyle='--', label='Baseline')
axs[0,0].set_title("Test Accuracy (%)", fontsize=13, fontweight="bold")
axs[0,0].set_xlabel("Epoch"); axs[0,0].set_ylabel("Accuracy (%)")
axs[0,0].legend(fontsize=10)
axs[0,0].set_ylim(0, 100)

# Top-right: Loss Components
axs[0,1].plot(df['Epoch'], df['Train_CLS_Loss'], 'g-s', linewidth=2, markersize=5, label="Classification Loss")
axs[0,1].plot(df['Epoch'], df['Train_SAC_Loss'], 'b-o', linewidth=2, markersize=5, label="SAC  Loss")
axs[0,1].plot(df['Epoch'], df['Test_Loss'], 'r--', linewidth=2, label="Test Loss")
axs[0,1].set_title("Loss Components", fontsize=13, fontweight="bold")
axs[0,1].set_xlabel("Epoch"); axs[0,1].set_ylabel("Loss")
axs[0,1].legend(fontsize=10)
axs[0,1].set_yscale('log')  # Optional: log scale for clarity

# Bottom-left: Time Analysis
axs[1,0].bar(df['Epoch'], df['Epoch_Time_s'], color="skyblue", alpha=0.7, label="Epoch Time")
axs[1,0].plot(df['Epoch'], df['Cumulative_Time_s']/60, 'k--', linewidth=2, label="Cumulative Time (min)")
axs[1,0].set_title("Epoch & Cumulative Training Time", fontsize=13, fontweight="bold")
axs[1,0].set_xlabel("Epoch"); axs[1,0].set_ylabel("Time (s/min)")
axs[1,0].legend(fontsize=10)

# Bottom-right: Confidence Threshold
axs[1,1].plot(df['Epoch'], df['Conf_Threshold'], 'm-^', linewidth=2, markersize=6, label="Confidence Threshold")
axs[1,1].set_title("Confidence Threshold", fontsize=13, fontweight="bold")
axs[1,1].set_xlabel("Epoch"); axs[1,1].set_ylabel("Confidence")
axs[1,1].legend(fontsize=10)
axs[1,1].set_ylim(0.75, 1.0)

plt.suptitle("SAC Training Analysis - Office-31 Dataset", fontsize=15, fontweight="bold")
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(f"{plot_dir}/training_analysis_grid.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{plot_dir}/training_analysis_grid.pdf", bbox_inches="tight")
plt.close()

# =========================
# 3. Summary Stats
# =========================
best_epoch = df['Test_Accuracy'].idxmax() + 1
best_acc = df['Test_Accuracy'].max()
total_time_min = df['Cumulative_Time_s'].iloc[-1]/60
avg_epoch_time = df['Epoch_Time_s'].mean()

print("=== SAC Training Summary ===")
print(f"Best Accuracy: {best_acc:.2f}% at Epoch {best_epoch}")
print(f"Final Test Accuracy: {df['Test_Accuracy'].iloc[-1]:.2f}%")
print(f"Baseline Accuracy: {baseline_acc}%")
print(f"Improvement: {best_acc - baseline_acc:+.2f}%")
print(f"Total Training Time: {total_time_min:.2f} min")
print(f"Average Epoch Time: {avg_epoch_time:.2f} s")
print(f"Confidence Threshold Range: {df['Conf_Threshold'].min():.3f} - {df['Conf_Threshold'].max():.3f}")
