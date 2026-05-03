import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from mmseg.apis import init_model, inference_model

# ==========================================
# 1. CONFIGURATION (Update these paths!)
# ==========================================
config_file = '/app/config.py'
checkpoint_file = '/app/output/best_mIoU_iter_17000_swin_t.pth'
img_path = '/app/data/2026-01-19-defect_dataset/images/fretting-bicycle_image_349.jpg'
output_path = '/app/analysis_dashboard.png'

class_names = ["bg", "cracks", "cracks_alligator", "cracks_severe", 
               "edge_cracks", "fretting", "pothole", "manhole", "pole_shadow"]

# Which class do you want to overlay? (6 = pothole, 1 = cracks, etc.)
target_class_idx = 6 

# ==========================================
# 2. INFERENCE & PROBABILITY EXTRACTION
# ==========================================
print(f"Loading model and analyzing {img_path.split('/')[-1]}...")
model = init_model(config_file, checkpoint_file, device='cuda:0')
result = inference_model(model, img_path)

# Extract logits and convert to probabilities
logits = result.seg_logits.data 
probabilities = torch.softmax(logits, dim=0).cpu().numpy()

# Calculate final mask and uncertainty
final_prediction = np.argmax(probabilities, axis=0)
max_probs = np.max(probabilities, axis=0) # High value = certain, Low value = uncertain

# ==========================================
# 3. VISUALIZATION DASHBOARD
# ==========================================
print("Generating visualizations...")

# Read original image via OpenCV (reads as BGR, convert to RGB for Matplotlib)
orig_img = cv2.imread(img_path)
orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

# Create a 2x2 grid of plots
fig, axs = plt.subplots(2, 2, figsize=(16, 12))

# --- Top Left: Original Image ---
axs[0, 0].imshow(orig_img)
axs[0, 0].set_title('Original Image', fontsize=14)
axs[0, 0].axis('off')

# --- Top Right: Final Segmentation Mask ---
mask_plot = axs[0, 1].imshow(final_prediction, cmap='nipy_spectral', vmin=0, vmax=8)
axs[0, 1].set_title('Final Segmentation Mask', fontsize=14)
axs[0, 1].axis('off')
fig.colorbar(mask_plot, ax=axs[0, 1], ticks=range(9), label='Class Index')

# --- Bottom Left: Confidence Heatmap Overlay ---
axs[1, 0].imshow(orig_img) # Show base image
# Overlay the heatmap with 50% transparency (alpha=0.5)
heatmap = axs[1, 0].imshow(probabilities[target_class_idx], cmap='jet', alpha=0.5, vmin=0, vmax=1)
axs[1, 0].set_title(f'{class_names[target_class_idx].capitalize()} Confidence Overlay', fontsize=14)
axs[1, 0].axis('off')
fig.colorbar(heatmap, ax=axs[1, 0], label='Probability (0 to 1)')

# --- Bottom Right: Model Uncertainty ---
# Using 'jet_r' (reversed jet): Red = highly uncertain (low max prob), Blue = very certain
uncert_plot = axs[1, 1].imshow(max_probs, cmap='jet_r', vmin=0, vmax=1)
axs[1, 1].set_title('Model Uncertainty (Red = Confused)', fontsize=14)
axs[1, 1].axis('off')
fig.colorbar(uncert_plot, ax=axs[1, 1], label='Max Probability per Pixel')

# ==========================================
# 4. SAVE RESULTS
# ==========================================
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Success! Analysis saved to: {output_path}")