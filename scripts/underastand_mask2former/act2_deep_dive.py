import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from mmseg.apis import init_model, inference_model

# ==========================================
# 1. CONFIGURATION
# ==========================================
config_file = '/app/config.py'
checkpoint_file = '/app/output/best_mIoU_iter_17000_swin_t.pth' 
img_path = '/app/data/2026-01-19-defect_dataset/images/fretting-bicycle_image_349.jpg'
output_path = '/app/blobs/act2_deep_dive.png'

class_names = np.array(["bg", "cracks", "cracks_alligator", "cracks_severe", 
                        "edge_cracks", "fretting", "pothole", "manhole", "pole_shadow"])
no_obj_idx = 9

print(f"Building 'The Lock and Key' Dashboard for: {img_path.split('/')[-1]}...")
model = init_model(config_file, checkpoint_file, device='cuda:0')

# ==========================================
# 2. THE WIRETAPS
# ==========================================
data = {}

def pixel_decoder_hook(module, args, output):
    # Safely hunt down the Master Map (We want the tensor with [Batch, Channels, H, W] where Channels is usually 256)
    if isinstance(output, tuple):
        for item in output:
            if isinstance(item, torch.Tensor) and item.dim() == 4:
                data['master_map'] = item[0].detach().cpu()
                break
        if 'master_map' not in data: # Fallback
            data['master_map'] = output[0][0].detach().cpu()
    else:
        data['master_map'] = output[0].detach().cpu()

def head_hook(module, args, output):
    cls_scores, mask_preds = output[0], output[1]
    if isinstance(cls_scores, list): cls_scores = torch.stack(cls_scores)
    if isinstance(mask_preds, list): mask_preds = torch.stack(mask_preds)
    data['cls_scores'] = cls_scores.detach().cpu() 
    data['mask_preds'] = mask_preds.detach().cpu()

h1 = model.decode_head.pixel_decoder.register_forward_hook(pixel_decoder_hook)
h2 = model.decode_head.register_forward_hook(head_hook)

result = inference_model(model, img_path)

h1.remove()
h2.remove()

# ==========================================
# 3. ISOLATE THE CHAMPION DEFECT
# ==========================================
all_cls = data['cls_scores'][:, 0] 
all_masks = data['mask_preds'][:, 0] 
master_map = data['master_map'] 
num_layers = all_cls.shape[0]

# Force the champion to be an actual defect (Ignore index 0 'bg' and index 9 'no_object')
final_probs = torch.softmax(all_cls[-1], dim=-1).numpy()
defect_probs = final_probs[:, 1:no_obj_idx] 
champion_idx = np.unravel_index(np.argmax(defect_probs), defect_probs.shape)[0]
champion_class_idx = np.argmax(final_probs[champion_idx][:no_obj_idx])
champion_name = class_names[champion_class_idx]

print(f"Champion Defect Found: Blob #{champion_idx} (Target: {champion_name.upper()})")

# Find the peak pixel of the champion's mask
final_mask = all_masks[-1, champion_idx].numpy()
max_y, max_x = np.unravel_index(np.argmax(final_mask), final_mask.shape)

# Get the Lock at this exact pixel (e.g., all 256 dimensions)
pixel_lock = master_map[:, max_y, max_x].numpy()
channels = pixel_lock.shape[0]

# Get the Dot Product history
dot_product_history = all_masks[:, champion_idx, max_y, max_x].numpy()

# ==========================================
# 4. DRAWING THE DASHBOARD
# ==========================================
orig_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
img_h, img_w = orig_img.shape[:2]

fig = plt.figure(figsize=(20, 12))
fig.suptitle(f"THE LOCK AND KEY: How Query #{champion_idx} uncovered {champion_name.upper()}", 
             fontsize=22, fontweight='bold')

# --- Panel 1: The Target Pixel ---
ax1 = plt.subplot2grid((3, 3), (0, 0), rowspan=2)
ax1.imshow(orig_img)
plot_x = int(max_x * (img_w / final_mask.shape[1]))
plot_y = int(max_y * (img_h / final_mask.shape[0]))
ax1.plot(plot_x, plot_y, marker='+', color='lime', markersize=30, markeredgewidth=4)
ax1.text(plot_x + 15, plot_y - 15, "Target Pixel", color='lime', fontsize=14, fontweight='bold', bbox=dict(facecolor='black', alpha=0.5))
ax1.set_title("1. The Target Pixel", fontsize=16, fontweight='bold')
ax1.axis('off')

# --- Panel 2: The Lock (Dynamic Barcode) ---
ax2 = plt.subplot2grid((3, 3), (0, 1), colspan=2)
# Dynamically create a perfect square grid for whatever number of channels we intercepted
grid_dim = int(np.ceil(np.sqrt(channels)))
lock_grid = np.zeros(grid_dim * grid_dim)
lock_grid[:channels] = pixel_lock
lock_grid = lock_grid.reshape(grid_dim, grid_dim)

im2 = ax2.imshow(lock_grid, cmap='viridis', aspect='auto')
ax2.set_title(f"2. The 'Lock' ({channels}-Dimensional Master Map Recipe at Target Pixel)", fontsize=16, fontweight='bold')
fig.colorbar(im2, ax=ax2, label="Feature Activation")
ax2.axis('off')

# --- Panel 3: The Key Turning (Graph) ---
ax3 = plt.subplot2grid((3, 3), (1, 1), colspan=2)
layers = np.arange(1, num_layers + 1)
ax3.plot(layers, dot_product_history, marker='o', markersize=10, color='royalblue', linewidth=3)
ax3.axhline(0, color='red', linestyle='--', linewidth=2, label="Threshold (Lock Opens)")
ax3.fill_between(layers, dot_product_history, 0, where=(dot_product_history > 0), color='lime', alpha=0.3, label="Unlocked (Inside Mask)")
ax3.fill_between(layers, dot_product_history, 0, where=(dot_product_history <= 0), color='salmon', alpha=0.3, label="Locked (Outside Mask)")
ax3.set_title("3. The 'Key' Turning (Dot Product Value across Transformer Layers)", fontsize=16, fontweight='bold')
ax3.set_xlabel("Transformer Layer (Time)", fontsize=14)
ax3.set_ylabel("Dot Product (Raw Logit)", fontsize=14)
ax3.set_xticks(layers)
ax3.legend(fontsize=12, loc='upper left')
ax3.grid(True, alpha=0.3)

# --- Panel 4: Masks ---
for i, layer_idx in enumerate([0, num_layers//2, num_layers-1]):
    ax = plt.subplot2grid((3, 3), (2, i))
    mask = all_masks[layer_idx, champion_idx].numpy()
    mask_sigmoid = 1.0 / (1.0 + np.exp(-mask))
    mask_resized = cv2.resize(mask_sigmoid, (img_w, img_h))
    
    ax.imshow(orig_img)
    overlay = np.zeros((img_h, img_w, 4), dtype=np.float32)
    overlay[mask_resized > 0.5] = [0.0, 1.0, 0.0, 0.6] 
    ax.imshow(overlay)
    
    logit_val = dot_product_history[layer_idx]
    status = "LOCKED" if logit_val < 0 else "UNLOCKED"
    ax.set_title(f"Layer {layer_idx + 1} Mask\nDot Product: {logit_val:.2f} ({status})", fontsize=14)
    ax.axis('off')

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Success! Lock and Key dashboard saved to: {output_path}")