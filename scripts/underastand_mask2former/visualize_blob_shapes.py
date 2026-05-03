import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from mmseg.apis import init_model, inference_model

# ==========================================
# 1. CONFIGURATION
# ==========================================
config_file = '/app/config.py'
checkpoint_file = '/app/output/best_mIoU_iter_17000_swin_t.pth' # UPDATE IF NEEDED
img_path = '/app/data/2026-01-19-defect_dataset/images/fretting-bicycle_image_349.jpg' # UPDATE IF NEEDED
output_img_path = '/app/blobs/blob_shapes_visualized.png'

class_names = ["bg", "cracks", "cracks_alligator", "cracks_severe", 
               "edge_cracks", "fretting", "pothole", "manhole", "pole_shadow"]
no_object_class_idx = 9

# ==========================================
# 2. INFERENCE & HOOK
# ==========================================
print(f"Extracting mask shapes for: {img_path.split('/')[-1]}...")
model = init_model(config_file, checkpoint_file, device='cuda:0')
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

# Get the data
final_cls_scores = blob_data['cls_scores'][-1, 0]
final_mask_preds = blob_data['mask_preds'][-1, 0]

blob_probabilities = torch.softmax(final_cls_scores, dim=-1).numpy()
# Get actual shapes (threshold at 0 for logits)
blob_masks_binary = (final_mask_preds.numpy() > 0).astype(np.uint8) 

num_blobs = blob_probabilities.shape[0]

# ==========================================
# 3. SORT BY CONFIDENCE
# ==========================================
# Calculate object confidence for all blobs
blob_records = []
for i in range(num_blobs):
    prob_vec = blob_probabilities[i]
    obj_conf = 1.0 - prob_vec[no_object_class_idx]
    pred_class_idx = np.argmax(prob_vec[:no_object_class_idx])
    
    blob_records.append({
        'id': i,
        'confidence': obj_conf,
        'class_name': class_names[pred_class_idx],
        'mask': blob_masks_binary[i]
    })

# Sort descending by confidence and take the Top 6
blob_records.sort(key=lambda x: x['confidence'], reverse=True)
top_blobs = blob_records[:6]

# ==========================================
# 4. PLOT THE SHAPES
# ==========================================
print("Generating shape visualizations...")
orig_img = cv2.imread(img_path)
orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
img_h, img_w, _ = orig_img.shape

fig, axs = plt.subplots(2, 3, figsize=(18, 12))
axs = axs.flatten()

for idx, blob in enumerate(top_blobs):
    ax = axs[idx]
    
    # 1. Show the original image as background
    ax.imshow(orig_img)
    
    # 2. Resize the 128x256 mask shape to match your original image resolution
    mask_resized = cv2.resize(blob['mask'], (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    
    # 3. Create a colored overlay for the mask
    # We create an RGBA image. Wherever the mask is 1, we color it Red with 50% opacity.
    overlay = np.zeros((img_h, img_w, 4), dtype=np.float32)
    overlay[mask_resized == 1] = [1.0, 0.0, 0.0, 0.5] # Red color, 0.5 Alpha
    
    # Draw the overlay
    ax.imshow(overlay)
    
    # Add titles and formatting
    ax.set_title(f"Rank {idx+1} (Blob #{blob['id']})\nClass: {blob['class_name'].upper()}\nConfidence: {blob['confidence']*100:.1f}%", 
                 fontsize=14, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.savefig(output_img_path, dpi=300, bbox_inches='tight')
print(f"✅ Success! Blob shapes visualized and saved to: {output_img_path}")