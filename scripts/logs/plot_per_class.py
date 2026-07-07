import math
import os
import matplotlib.pyplot as plt

# =====================================================================
# CONFIGURATION
# =====================================================================

MODELS = {
    'Swin': 'logs_training/log_swin_t.log',
    'BeiT': 'logs_training/log_beit_b.log',
    'HRNet': 'logs_training/log_hrnet_t.log',
    'InterImage': 'logs_training/log_intern_t.log',
    'FlashInterImage': 'logs_training/log_flash_t.log'
}

ITER_PER_EPOCH = 725.0
TARGET_CLASSES = ['cracks', 'cracks_severe', 'cracks_alligator', 'pothole'] 

# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def smooth_curve(scalars, weight=0.6):
    """
    Applies Exponential Moving Average smoothing.
    Weight between 0 and 1. Higher weight = smoother curve.
    """
    if not scalars: return []
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def parse_log(log_file):
    """Extracts epochs and class IoUs from a single log file."""
    epochs = []
    class_ious = {}
    
    if not os.path.exists(log_file):
        print(f"  [!] Warning: Could not find {log_file}. Skipping.")
        return epochs, class_ious

    with open(log_file, 'r') as f:
        lines = f.readlines()

    in_table = False
    table_idx = 0

    for line in lines:
        if '|      Class       |' in line:
            in_table = True
            table_idx += 1
            epochs.append((table_idx * 1000) / ITER_PER_EPOCH)
            continue
            
        if in_table:
            if 'Iter(val)' in line and 'aAcc' in line:
                in_table = False
                continue
                
            if '+----' in line:
                continue
                
            parts = [p.strip() for p in line.split('|')]
            if len(parts) > 3 and parts[1] != 'Class' and parts[1] != '':
                cls_name = parts[1]
                try:
                    iou = float(parts[2])
                except ValueError:
                    iou = 0.0 
                    
                if cls_name not in class_ious:
                    class_ious[cls_name] = []
                class_ious[cls_name].append(iou)

    return epochs, class_ious

# --- 1. Gather Data ---
print("Parsing logs for all models...")
all_data = {}
all_available_classes = set()

for model_name, log_path in MODELS.items():
    print(f"-> Processing {model_name}...")
    epochs, class_ious = parse_log(log_path)
    if epochs:
        all_data[model_name] = {'epochs': epochs, 'class_ious': class_ious}
        all_available_classes.update(class_ious.keys())

if not all_data:
    print("Error: No data extracted from any log files. Exiting.")
    exit()

# --- 2. Determine which classes to plot ---
all_available_classes.discard('bg')

if TARGET_CLASSES:
    classes_to_plot = [c for c in TARGET_CLASSES if c in all_available_classes]
else:
    classes_to_plot = sorted(list(all_available_classes))

n_classes = len(classes_to_plot)
if n_classes == 0:
    print("Error: No valid classes to plot.")
    exit()

print(f"\nPlotting {n_classes} classes: {', '.join(classes_to_plot)}")

# --- 3. Dynamic Grid Setup ---
ncols = min(2, n_classes)
nrows = math.ceil(n_classes / ncols)

# Adjusted figsize to make room for the global legend at the bottom
fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4 * nrows), sharex=True, sharey=True)

if n_classes == 1:
    axes = [axes]
else:
    axes = axes.flatten()

model_colors = ['#2F4F4F', '#FF1493', '#00CED1', '#FFA500', '#4B0082']

# --- 4. Plotting ---
for idx, cls_name in enumerate(classes_to_plot):
    ax = axes[idx]
    
    for model_idx, (model_name, data) in enumerate(all_data.items()):
        if cls_name in data['class_ious']:
            epochs = data['epochs']
            ious = data['class_ious'][cls_name]
            color = model_colors[model_idx % len(model_colors)]
            
            min_len = min(len(epochs), len(ious))
            ep_trunc = epochs[:min_len]
            iou_trunc = ious[:min_len]
            
            # (DELETED the faint raw data line here)
            
            # Plot ONLY the bold, smoothed trendline
            smoothed_ious = smooth_curve(iou_trunc, weight=0.4)
            ax.plot(ep_trunc, smoothed_ious, linewidth=1.5, color=color, label=model_name)
    
    ax.set_title(cls_name, fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    if idx >= (nrows - 1) * ncols:
        ax.set_xlabel('Epochs')
    if idx % ncols == 0:
        ax.set_ylabel('IoU (%)')

for i in range(n_classes, len(axes)):
    fig.delaxes(axes[i])

# --- 5. Global Legend & Save Output ---
# Extract handles from the first axis to create a single global legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=len(MODELS), bbox_to_anchor=(0.5, -0.02), frameon=False)

# Adjust layout to prevent the title or legend from being cut off
fig.suptitle('Model Comparison: Per-Class Validation IoU', fontsize=16, y=1.02)
plt.tight_layout()
# Add a bit of extra space at the bottom for the legend
plt.subplots_adjust(bottom=0.1)

output_name = f'Model_Comparison_{n_classes}_classes_smoothed.png'
plt.savefig(output_name, dpi=300, bbox_inches='tight')
print(f"Success! Smoothed graph saved to {output_name}")