import re
import matplotlib.pyplot as plt

# Update this path to wherever your log file is!
log_file = 'logs_training/log_beit_b.log' 
ITER_PER_EPOCH = 725.0

epochs = []
class_ious = {}

print("Parsing per-class validation tables...")

try:
    with open(log_file, 'r') as f:
        lines = f.readlines()

    in_table = False
    table_idx = 0

    for line in lines:
        # 1. Detect the header of the validation table
        if '|      Class       |' in line:
            in_table = True
            table_idx += 1
            epochs.append((table_idx * 1000) / ITER_PER_EPOCH)
            continue
            
        if in_table:
            # 2. Stop reading when we hit the summary line below the table
            if 'Iter(val)' in line and 'aAcc' in line:
                in_table = False
                continue
                
            # 3. Ignore the +---+ separator lines
            if '+----' in line:
                continue
                
            # 4. Extract the data!
            parts = [p.strip() for p in line.split('|')]
            
            if len(parts) > 3 and parts[1] != 'Class' and parts[1] != '':
                cls_name = parts[1]
                iou_str = parts[2]
                
                try:
                    iou = float(iou_str)
                except ValueError:
                    iou = 0.0 # Handle 'nan' safely
                    
                if cls_name not in class_ious:
                    class_ious[cls_name] = []
                class_ious[cls_name].append(iou)

    print(f"Successfully extracted data for {len(class_ious)} classes over {len(epochs)} validation steps.")

    # --- Plotting (The Small Multiples Fix) ---
    
    # Create a 2x4 grid of subplots. 
    # sharex and sharey ensure all graphs use the exact same scale!
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True, sharey=True)
    axes = axes.flatten() # Flattens the 2D array of axes into a 1D list for easy looping
    
    # A list of distinct, professional colors
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#f781bf', '#666666']
    
    # Filter out the background class
    classes_to_plot = [cls for cls in class_ious.keys() if cls != 'bg']
    
    for idx, cls_name in enumerate(classes_to_plot):
        ax = axes[idx]
        ious = class_ious[cls_name]
        color = colors[idx % len(colors)]
        
        # Plot the line with a gentle marker
        ax.plot(epochs, ious, marker='o', markersize=3, linewidth=2, color=color)
        
        # Add a faint trendline (optional, but looks very academic)
        # smoothed = smooth_curve(ious, weight=0.7) # Uncomment if you define smooth_curve above!
        # ax.plot(epochs, smoothed, color=color, linewidth=2) 
        
        # Format each mini-graph
        ax.set_title(cls_name, fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Only add X labels to the bottom row, and Y labels to the left column
        if idx >= 4:
            ax.set_xlabel('Epochs')
        if idx % 4 == 0:
            ax.set_ylabel('IoU (%)')

    # Add a main title for the whole figure
    fig.suptitle('BeiT v2: Per-Class Validation IoU', fontsize=16, y=1.02)
    
    plt.tight_layout()
    output_name = 'Per_Class_IoU_Grid.png'
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"Success! Grid graph saved to {output_name}")

except FileNotFoundError:
    print(f"Error: Could not find {log_file}.")