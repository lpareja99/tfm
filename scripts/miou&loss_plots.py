import re
import matplotlib.pyplot as plt

# --- ADD THIS VARIABLE ---
ITER_PER_EPOCH = 725.0 

experiments = {
    'FlashInternImage': 'logs_training/log_flash_t.log',
    'HRNet': 'logs_training/log_hrnet_t.log',
    'Swin': 'logs_training/log_swin_t.log',
    'BeiT2': 'logs_training/log_beit_b.log',
    'InternImage': 'logs_training/log_intern_t.log'
}

def smooth_curve(scalars, weight=0.96): 
    if not scalars: return []
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00'] 

print("Parsing log files...")

for idx, (label, log_file) in enumerate(experiments.items()):
    train_epochs, train_losses = [], []  # Renamed variable
    val_epochs, val_mious = [], []       # Renamed variable
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                if 'Iter(train)' in line and 'loss:' in line:
                    match_iter = re.search(r'\[\s*(\d+)/', line)
                    match_loss = re.search(r'loss:\s*([0-9.]+)', line)
                    if match_iter and match_loss:
                        # --- DIVIDE BY 725 HERE ---
                        current_epoch = int(match_iter.group(1)) / ITER_PER_EPOCH
                        train_epochs.append(current_epoch)
                        train_losses.append(float(match_loss.group(1)))
                        
                elif 'Iter(val)' in line and 'mIoU:' in line:
                    match_miou = re.search(r'mIoU:\s*([0-9.]+)', line)
                    if match_miou:
                        # --- DIVIDE BY 725 HERE ---
                        current_epoch = int(match_iter.group(1)) / ITER_PER_EPOCH
                        val_epochs.append(current_epoch)
                        val_mious.append(float(match_miou.group(1)))
        
        current_color = colors[idx % len(colors)]
        
        smoothed_loss = smooth_curve(train_losses, weight=0.96)
        # Update plotting variables
        ax1.plot(train_epochs, smoothed_loss, color=current_color, linestyle='-', linewidth=1.5, label=label)
        ax2.plot(val_epochs, val_mious, marker='o', markersize=3, linestyle='-', color=current_color, label=label)
        
    except FileNotFoundError:
        print(f"Warning: Could not find {log_file}. Skipping {label}.")

ax1.set_title('Training Loss (Smoothed)')
ax1.set_xlabel('Epochs')  # --- CHANGE LABEL ---
ax1.set_ylabel('Loss')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()

ax2.set_title('Validation mIoU')
ax2.set_xlabel('Epochs')  # --- CHANGE LABEL ---
ax2.set_ylabel('mIoU (%)')
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend()

plt.tight_layout()
output_name = 'output/Model_Comparison_Epochs.png'
plt.savefig(output_name, dpi=300, bbox_inches='tight')
print(f"Success! Comparison graph saved as {output_name}")