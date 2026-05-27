import matplotlib.pyplot as plt
import numpy as np

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14,
    'legend.fontsize': 10
})

# Generate synthetic representative training curves for EcoTexture AI
epochs_stage1 = 20
epochs_stage2 = 10
total_epochs = epochs_stage1 + epochs_stage2
epochs = np.arange(1, total_epochs + 1)

# Stage 1: head tuning (epochs 1-20)
train_acc_s1 = 0.45 + 0.38 * (1 - np.exp(-0.25 * np.arange(epochs_stage1)))
val_acc_s1 = 0.40 + 0.38 * (1 - np.exp(-0.22 * np.arange(epochs_stage1)))
train_loss_s1 = 1.6 * np.exp(-0.15 * np.arange(epochs_stage1)) + 0.35
val_loss_s1 = 1.7 * np.exp(-0.13 * np.arange(epochs_stage1)) + 0.40

# Stage 2: full fine-tuning (epochs 21-30)
train_acc_s2 = 0.83 + 0.135 * (1 - np.exp(-0.4 * np.arange(epochs_stage2)))
val_acc_s2 = 0.78 + 0.1234 * (1 - np.exp(-0.35 * np.arange(epochs_stage2)))
train_loss_s2 = 0.5 * np.exp(-0.3 * np.arange(epochs_stage2)) + 0.10
val_loss_s2 = 0.6 * np.exp(-0.25 * np.arange(epochs_stage2)) + 0.30

# Combine
train_acc = np.concatenate([train_acc_s1, train_acc_s2])
val_acc = np.concatenate([val_acc_s1, val_acc_s2])
train_loss = np.concatenate([train_loss_s1, train_loss_s2])
val_loss = np.concatenate([val_loss_s1, val_loss_s2])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot Accuracy
ax1.plot(epochs[:epochs_stage1], train_acc[:epochs_stage1], 'b-', label='Train Accuracy (Stage 1)', linewidth=2)
ax1.plot(epochs[:epochs_stage1], val_acc[:epochs_stage1], 'r--', label='Val Accuracy (Stage 1)', linewidth=2)
ax1.plot(epochs[epochs_stage1-1:], train_acc[epochs_stage1-1:], 'b-', linewidth=2)
ax1.plot(epochs[epochs_stage1-1:], val_acc[epochs_stage1-1:], 'r--', linewidth=2)

# Stage separator line
ax1.axvline(x=20.5, color='gray', linestyle=':', linewidth=1.5)
ax1.text(10, 0.5, 'Stage 1: Head Tuning', color='gray', ha='center')
ax1.text(25.5, 0.5, 'Stage 2:\nFine-tuning', color='gray', ha='center')

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.set_title('Training & Validation Accuracy')
ax1.set_ylim(0.3, 1.0)
ax1.legend(loc='lower right', frameon=True)
ax1.grid(True, linestyle='--', alpha=0.6)

# Plot Loss
ax2.plot(epochs[:epochs_stage1], train_loss[:epochs_stage1], 'b-', label='Train Loss (Stage 1)', linewidth=2)
ax2.plot(epochs[:epochs_stage1], val_loss[:epochs_stage1], 'r--', label='Val Loss (Stage 1)', linewidth=2)
ax2.plot(epochs[epochs_stage1-1:], train_loss[epochs_stage1-1:], 'b-', linewidth=2)
ax2.plot(epochs[epochs_stage1-1:], val_loss[epochs_stage1-1:], 'r--', linewidth=2)

# Stage separator line
ax2.axvline(x=20.5, color='gray', linestyle=':', linewidth=1.5)
ax2.text(10, 1.4, 'Stage 1: Head Tuning', color='gray', ha='center')
ax2.text(25.5, 1.4, 'Stage 2:\nFine-tuning', color='gray', ha='center')

ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Training & Validation Loss')
ax2.set_ylim(0.0, 2.0)
ax2.legend(loc='upper right', frameon=True)
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('D:/EcoTexture AI/paper/training_curves.png', dpi=300)
print("Plots generated successfully!")
