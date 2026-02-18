import csv
import re
import argparse
import os
import matplotlib.pyplot as plt

def parse_training_log(log_path, out_dir):
    # Patterns for training metadata
    train_pattern = re.compile(r'grad_norm:\s+([\d\.]+).*loss:\s+([\d\.]+)')
    ckpt_pattern = re.compile(r'Saving checkpoint at (\d+) iterations')
    table_border = "+--------+-------+-------+-------+--------+-----------+--------+"
    
    results = []
    last_grad, last_loss, current_iter = "N/A", "N/A", "N/A"
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # 1. Capture latest training stats
            if 'Iter(train)' in line:
                train_match = train_pattern.search(line)
                if train_match:
                    last_grad = train_match.group(1)
                    last_loss = train_match.group(2)

            # 2. Capture specific iteration from checkpoint save
            ckpt_match = ckpt_pattern.search(line)
            if ckpt_match:
                current_iter = ckpt_match.group(1)

            # 3. Process the entire results table
            if table_border in line and (i + 1) < len(lines) and "Class" in lines[i+1]:
                # Skip the border and header to get to the first class row
                i += 3 
                
                # Loop through rows until we hit the bottom border
                while i < len(lines) and table_border not in lines[i]:
                    row_content = [val.strip() for val in lines[i].split('|')]
                    
                    if len(row_content) >= 8:
                        results.append({
                            'Iteration': int(current_iter),
                            'Loss': float(last_loss) if last_loss != "N/A" else 0.0,
                            'Grad_Norm': float(last_grad) if last_grad != "N/A" else 0.0,
                            'Class': row_content[1], # e.g., 'bg' or 'cracks'
                            'IoU': float(row_content[2]) if row_content[2] != "N/A" else 0.0,
                            'Acc': float(row_content[3]) if row_content[3] != "N/A" else 0.0,
                            'Dice': float(row_content[4]) if row_content[4] != "N/A" else 0.0,
                            'Fscore': float(row_content[5]) if row_content[5] != "N/A" else 0.0,
                            'Precision': float(row_content[6]) if row_content[6] != "N/A" else 0.0,
                            'Recall': float(row_content[7]) if row_content[7] != "N/A" else 0.0
                        })
                    i += 1
            i += 1

    if not results:
        print("⚠ No data found in the log.")
        return

    keys = results[0].keys()
    output_csv = os.path.join(out_dir, 'training_history.csv')
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✔ Saved {len(results)} class-specific records to {output_csv}")
    
    return results
    
    
def create_plot(results, out_dir):
    if not results:
        return
    
    classes = sorted(list(set(r['Class'] for r in results)))
    metrics = ['IoU', 'Acc', 'Precision', 'Recall']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Training Performance Metrics per Class", fontsize=16)
    
    # Flatten axes for easy iteration
    axes = axes.ravel()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for cls in classes:
            # Filter data for this specific class
            cls_data = [r for r in results if r['Class'] == cls]
            iters = [r['Iteration'] for r in cls_data]
            values = [r[metric] for r in cls_data]
            ax.plot(iters, values, marker='o', label=f"Class: {cls}")
        
        ax.set_title(f"Per-Class {metric} over Iterations")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Percentage (%)")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(out_dir, 'metrics_plot.png')
    plt.savefig(plot_path)
    print(f"✔ Composite plot saved to {plot_path}")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('log_path', help='Path to the .log file')
    parser.add_argument('out_dir', help='Output directory for plots')
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    data = parse_training_log(args.log_path, args.out_dir)
    create_plot(data, args.out_dir)