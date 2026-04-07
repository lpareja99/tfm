import csv
import re
import os
import argparse
import matplotlib.pyplot as plt

def parse_logs(log_path, out_dir):
    # Regex to find training loss (handles timestamps)
    # Example: ... Iter(train) [ 100/12500] ... loss: 33.9626
    train_pattern = re.compile(r'Iter\(train\).*?loss:\s+([\d\.]+)')
    
    # Regex to find validation table rows
    # Example: |      cracks      | 12.08 | ...
    row_pattern = re.compile(r'\|\s*(\w+)\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)\s*\|')

    results = []
    current_iter = 0
    
    # Read the file
    with open(log_path, 'r') as f:
        lines = f.readlines()

    print(f"Scanning {len(lines)} lines in {log_path}...")

    for i, line in enumerate(lines):
        # 1. Capture current iteration from training logs
        if 'Iter(train)' in line:
            # Extract iteration number [ 100/12500]
            iter_match = re.search(r'\[\s*(\d+)/\d+\]', line)
            if iter_match:
                current_iter = int(iter_match.group(1))
        
        # 2. Capture Validation Table Data
        # We look for lines containing "Class" and "|" to identify the start of a table
        if '|' in line and 'Class' in line and 'IoU' in line:
            # We found a header, now scan the next few lines for data
            for j in range(1, 10): # Scan next 10 lines max
                if i + j >= len(lines): break
                
                next_line = lines[i + j]
                match = row_pattern.search(next_line)
                
                if match:
                    class_name, iou, acc, dice = match.groups()
                    results.append({
                        'Iteration': current_iter,
                        'Class': class_name,
                        'IoU': float(iou),
                        'Acc': float(acc),
                        'Dice': float(dice)
                    })
                    # print(f"Found data: Iter {current_iter} - {class_name} IoU: {iou}")

    # Save to CSV
    if results:
        csv_file = os.path.join(out_dir, 'training_metrics.csv')
        keys = results[0].keys()
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"✅ Saved CSV to {csv_file}")
        return results
    else:
        print("❌ No data found! Check log format.")
        return []

def plot_metrics(results, out_dir):
    if not results: return

    # Get unique classes
    classes = sorted(list(set(r['Class'] for r in results)))
    metrics = ['IoU', 'Acc', 'Dice']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for cls in classes:
            # Filter data for this class
            cls_data = [r for r in results if r['Class'] == cls]
            # Sort by iteration to ensure correct line plotting
            cls_data.sort(key=lambda x: x['Iteration'])
            
            x = [r['Iteration'] for r in cls_data]
            y = [r[metric] for r in cls_data]
            
            ax.plot(x, y, marker='o', label=cls)
        
        ax.set_title(f'{metric} over Training')
        ax.set_xlabel('Iteration')
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        if idx == 0: ax.legend() # Only show legend on first plot

    plot_path = os.path.join(out_dir, 'metrics_plot.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"✅ Saved Plot to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file", help="Path to std_log.txt")
    parser.add_argument("out_dir", help="Folder to save results")
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    data = parse_logs(args.log_file, args.out_dir)
    plot_metrics(data, args.out_dir)