import csv
import re
import os
import matplotlib.pyplot as plt

def is_table_border(line):
    # Matches a line like +-------+-------+...
    return re.match(r'^\+[-+]+\+$', line.strip()) is not None

def parse_training_log(log_path, out_dir, run_name=""):
    train_pattern = re.compile(r'grad_norm:\s+([\d\.]+).*loss:\s+([\d\.]+)')
    ckpt_pattern = re.compile(r'Saving checkpoint at (\d+) iterations')
    
    results = []
    last_grad, last_loss, current_iter = "0.0", "0.0", "0"
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if 'Iter(train)' in line:
                m = train_pattern.search(line)
                if m: last_grad, last_loss = m.groups()
            m = ckpt_pattern.search(line)
            if m: current_iter = m.group(1)
            
            if is_table_border(line) and (i + 1) < len(lines) and "Class" in lines[i+1]:
                i += 3 
                while i < len(lines) and not is_table_border(lines[i]):
                    row = [val.strip() for val in lines[i].split('|')]
                    if len(row) >= 8:
                        results.append({
                            'Iteration': int(current_iter), 'Loss': float(last_loss),
                            'Grad_Norm': float(last_grad), 'Class': row[1],
                            'IoU': float(row[2]), 'Acc': float(row[3]),
                            'Precision': float(row[6]), 'Recall': float(row[7])
                        })
                    i += 1
    
    if results:
        # Use run_name to create a unique filename
        filename = f'training_history_{run_name}.csv' if run_name else 'training_history.csv'
        output_csv = os.path.join(out_dir, filename)
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"✔ Saved {len(results)} records to {filename}")
    return results


def parse_testing_log(log_path, out_dir, run_name=""):
    # Pattern to find which checkpoint was loaded
    ckpt_loaded_pattern = re.compile(r'Load checkpoint from (.*\.pth)')
    
    results = []
    checkpoint_used = "Unknown"
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            # 1. Capture the checkpoint source
            ckpt_match = ckpt_loaded_pattern.search(line)
            if ckpt_match:
                checkpoint_used = os.path.basename(ckpt_match.group(1))

            # 2. Extract the final table
            if is_table_border(line) and (i + 1) < len(lines) and "Class" in lines[i+1]:
                i += 3 
                while i < len(lines) and not is_table_border(lines[i]):
                    row = [val.strip() for val in lines[i].split('|')]
                    if len(row) >= 8:
                        results.append({
                            'Checkpoint': checkpoint_used,
                            'Class': row[1],
                            'IoU': float(row[2]),
                            'Acc': float(row[3]),
                            'Dice': float(row[4]),
                            'Precision': float(row[6]),
                            'Recall': float(row[7])
                        })
                    i += 1
    
    if results:
        # Save unique CSV
        filename = f'test_summary_{run_name}.csv'
        output_csv = os.path.join(out_dir, filename)
        keys = results[0].keys()
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"✔ Saved Test Summary to {filename}")
    return results


def create_plot(results, out_dir, run_name=""):
    if not results: return
    metrics = ['IoU', 'Acc', 'Precision', 'Recall']
    classes = sorted(list(set(r['Class'] for r in results)))
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Metrics for Run: {run_name}", fontsize=16)
    
    for idx, metric in enumerate(metrics):
        ax = axes.ravel()[idx]
        for cls in classes:
            cls_data = [r for r in results if r['Class'] == cls]
            ax.plot([r['Iteration'] for r in cls_data], [r[metric] for r in cls_data], marker='o', label=cls)
        ax.set_title(f"Per-Class {metric}")
        ax.set_ylim(0, 105)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Use run_name for the plot filename
    plot_filename = f'metrics_plot_{run_name}.png' if run_name else 'metrics_plot.png'
    plt.savefig(os.path.join(out_dir, plot_filename))
    plt.close()