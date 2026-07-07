import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import Rectangle
from mmseg.apis import init_model, inference_model

# ==========================================
# 1. CONFIGURATION
# ==========================================
config_file = '/app/config.py'
checkpoint_file = '/app/output/best_mIoU_iter_17000_swin_t.pth' # UPDATE
img_path = '/app/data/2026-01-19-defect_dataset/images/fretting-bicycle_image_349.jpg'
output_img_path = '/app/blobs/battle_aftermath.png'

class_names = np.array(["bg", "cracks", "cracks_alligator", "cracks_severe", 
                        "edge_cracks", "fretting", "pothole", "manhole", "pole_shadow"])

print(f"Analyzing battle aftermath for: {img_path.split('/')[-1]}...")
model = init_model(config_file, checkpoint_file, device='cuda:0')

# ==========================================
# 2. INTERCEPT DATA
# ==========================================
blob_data = {}
def mask2former_hook(module, args, output):
    cls_scores, mask_preds = output[0], output[1]
    if isinstance(cls_scores, list): cls_scores = torch.stack(cls_scores)
    if isinstance(mask_preds, list): mask_preds = torch.stack(mask_preds)
    blob_data['cls_scores'] = cls_scores.detach().cpu()
    blob_data['mask_preds'] = mask_preds.detach().cpu()

hook = model.decode_head.register_forward_hook(mask2former_hook)
result = inference_model(model, img_path)
hook.remove()

# Get raw probabilities
logits = result.seg_logits.data
probs = torch.softmax(logits, dim=0).cpu().numpy() # Shape: (9, H, W)
final_seg_map = result.pred_sem_seg.data[0].cpu().numpy() # Shape: (H, W)

# Get blob data
final_cls_scores = blob_data['cls_scores'][-1, 0]
final_mask_preds = blob_data['mask_preds'][-1, 0]
blob_probs = torch.softmax(final_cls_scores, dim=-1).numpy()
blob_masks = torch.sigmoid(final_mask_preds).numpy() # Use sigmoid for mask confidence

# ==========================================
# 3. FIND THE BATTLES
# ==========================================
sorted_probs = np.sort(probs, axis=0)
margin_of_victory = sorted_probs[-1, :, :] - sorted_probs[-2, :, :]
winning_class_idx = np.argmax(probs, axis=0)
margin_of_victory[winning_class_idx == 0] = 1.0 # Ignore background battles

flat_indices = np.argsort(margin_of_victory.flatten())[:3] # Let's do Top 3 to save space
battle_coords = [np.unravel_index(idx, margin_of_victory.shape) for idx in flat_indices]

# ==========================================
# 4. DASHBOARD GENERATION
# ==========================================
print("Generating the Aftermath Report...")
orig_img = cv2.imread(img_path)
orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
img_h, img_w = orig_img.shape[:2]

fig = plt.figure(figsize=(18, 12))
fig.suptitle("The Battle Aftermath: How the Model Made its Final Choice", fontsize=18, fontweight='bold')

# Top Row: Original Image with crosshairs
ax_main = plt.subplot2grid((4, 3), (0, 0), colspan=3)
ax_main.imshow(orig_img)
ax_main.axis('off')
colors = ['red', 'cyan', 'lime']

for i, (y, x) in enumerate(battle_coords):
    ax_main.plot(x, y, marker='+', markersize=20, color=colors[i], markeredgewidth=3)
    ax_main.text(x + 10, y - 10, f"Battle {i+1}", color=colors[i], fontsize=12, fontweight='bold', bbox=dict(facecolor='black', alpha=0.5))

    # --- Calculations for this pixel ---
    pixel_probs = probs[:, y, x]
    final_class = winning_class_idx[y, x]
    final_class_name = class_names[final_class]
    
    # Find which blob contributed the most to this winning class at this exact pixel
    # Scale coordinates to blob mask size (128x256)
    mask_y = int(y * (blob_masks.shape[1] / img_h))
    mask_x = int(x * (blob_masks.shape[2] / img_w))
    
    blob_contributions = blob_probs[:, final_class] * blob_masks[:, mask_y, mask_x]
    winning_blob_idx = np.argmax(blob_contributions)
    
    # Resize the winning blob's mask for visualization
    winning_blob_mask = cv2.resize(blob_masks[winning_blob_idx], (img_w, img_h))

    # --- Row setup for this battle ---
    row = i + 1
    
    # Col 1: The Probability Bar Chart
    ax_bar = plt.subplot2grid((4, 3), (row, 0))
    top_3_indices = np.argsort(pixel_probs)[-3:][::-1]
    top_3_probs = pixel_probs[top_3_indices]
    top_3_names = class_names[top_3_indices]
    bars = ax_bar.barh(top_3_names[::-1], top_3_probs[::-1], color=colors[i], alpha=0.7)
    ax_bar.set_title(f"Battle {i+1} Votes at ({x}, {y})", fontweight='bold')
    ax_bar.set_xlim(0, 1.0)
    for bar in bars:
        width = bar.get_width()
        ax_bar.text(width + 0.02, bar.get_y() + bar.get_height()/2, f'{width*100:.1f}%', va='center')
        
    # Col 2: The Final Predicted Patch (Zoomed 100x100)
    ax_patch = plt.subplot2grid((4, 3), (row, 1))
    zoom_size = 50
    y_min, y_max = max(0, y-zoom_size), min(img_h, y+zoom_size)
    x_min, x_max = max(0, x-zoom_size), min(img_w, x+zoom_size)
    
    patch_img = orig_img[y_min:y_max, x_min:x_max]
    patch_mask = final_seg_map[y_min:y_max, x_min:x_max]
    
    ax_patch.imshow(patch_img)
    ax_patch.imshow(patch_mask, cmap='nipy_spectral', vmin=0, vmax=8, alpha=0.5) # Overlay final mask
    ax_patch.plot(zoom_size, zoom_size, marker='+', color='white', markersize=15, markeredgewidth=2)
    ax_patch.set_title(f"Final Class Assigned:\n{final_class_name.upper()}", fontweight='bold')
    ax_patch.axis('off')

    # Col 3: The Winning Blob's Mask Overlay
    ax_blob = plt.subplot2grid((4, 3), (row, 2))
    ax_blob.imshow(orig_img)
    overlay = np.zeros((img_h, img_w, 4), dtype=np.float32)
    overlay[winning_blob_mask > 0.5] = [1.0, 1.0, 1.0, 0.5] # White overlay for the winning blob
    ax_blob.imshow(overlay)
    
    # Zoom into the blob region
    ax_blob.set_xlim(x_min-100, x_max+100)
    ax_blob.set_ylim(y_max+100, y_min-100)
    ax_blob.plot(x, y, marker='+', color=colors[i], markersize=15, markeredgewidth=2)
    ax_blob.set_title(f"Deciding Factor:\nBlob #{winning_blob_idx} Championed This Class", fontweight='bold')
    ax_blob.axis('off')

plt.tight_layout()
plt.savefig(output_img_path, dpi=300, bbox_inches='tight')
print(f"✅ Success! Report saved to: {output_img_path}")