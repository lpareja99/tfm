import re
import matplotlib.pyplot as plt

log_file = 'logs_training/log_beit_b.log' # Update path if needed
ITER_PER_EPOCH = 725.0

epochs = []
loss_cls = []
loss_mask = []
loss_dice = []

# Smoothing function
def smooth_curve(scalars, weight=0.8): 
    if not scalars: return []
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

print("Extracting detailed losses...")

try:
    with open(log_file, 'r') as f:
        for line in f:
            if 'Iter(train)' in line:
                # Extract Iteration
                match_iter = re.search(r'\[\s*(\d+)/', line)
                
                # Extract the 3 main losses
                match_cls = re.search(r'decode\.loss_cls:\s*([0-9.]+)', line)
                match_mask = re.search(r'decode\.loss_mask:\s*([0-9.]+)', line)
                match_dice = re.search(r'decode\.loss_dice:\s*([0-9.]+)', line)
                
                if match_iter and match_cls and match_mask and match_dice:
                    current_epoch = int(match_iter.group(1)) / ITER_PER_EPOCH
                    epochs.append(current_epoch)
                    
                    loss_cls.append(float(match_cls.group(1)))
                    loss_mask.append(float(match_mask.group(1)))
                    loss_dice.append(float(match_dice.group(1)))

    print(f"Found {len(epochs)} data points.")

    # Plotting
    plt.figure(figsize=(10, 6))
    
    plt.plot(epochs, smooth_curve(loss_cls), label='Classification Loss (What is it?)', color='#377eb8', linewidth=2)
    plt.plot(epochs, smooth_curve(loss_mask), label='Mask Loss (Pixel-level)', color='#4daf4a', linewidth=2)
    plt.plot(epochs, smooth_curve(loss_dice), label='Dice Loss (Shape/Boundaries)', color='#e41a1c', linewidth=2)

    plt.title('BeiT v2: Learning Objectives')
    plt.xlabel('Epochs')
    plt.ylabel('Smoothed Loss Value')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.savefig('output/Detailed_Loss_Types.png', dpi=300)
    print("Success! Graph saved to output/Detailed_Loss_Types.png")

except FileNotFoundError:
    print(f"Could not find the log file at {log_file}")