import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from mmseg.apis import init_model, inference_model

# ==========================================
# 1. CONFIGURATION
# ==========================================
config_file = '/app/config.py'
# UPDATE THIS PATH
checkpoint_file = '/app/output/best_mIoU_iter_17000_swin_t.pth'
img_path = '/app/data/2026-01-19-defect_dataset/images/fretting-bicycle_image_349.jpg'
output_img_path = '/app/blob_correlation_map.png'
output_pkl_path = '/app/blob_data.pkl'
output_csv_path = '/app/blob_data.csv'

class_names = ["bg", "cracks", "cracks_alligator", "cracks_severe", 
               "edge_cracks", "fretting", "pothole", "manhole", "pole_shadow"]
no_object_class_idx = 9
visualization_threshold = 0.01 

palette = [
    [0, 0, 0], [250, 50, 83], [36, 179, 83], [102, 204, 255], [255, 165, 0],
    [128, 128, 128], [255, 255, 0], [0, 255, 255], [255, 0, 255]
]
point_colors = np.array(palette) / 255.0 

# ==========================================
# 2. INFERENCE & HOOK
# ==========================================
print(f"Running inference on: {img_path.split('/')[-1]}...")
model = init_model(config_file, checkpoint_file, device='cuda:0')

blob_data = {}

def mask2former_hook(module, args, output):
    cls_scores = output[0]
    mask_preds = output[1]
    
    if isinstance(cls_scores, list): cls_scores = torch.stack(cls_scores)
    if isinstance(mask_preds, list): mask_preds = torch.stack(mask_preds)
        
    blob_data['cls_scores'] = cls_scores.detach().cpu()
    blob_data['mask_preds'] = mask_preds.detach().cpu()

hook = model.decode_head.register_forward_hook(mask2former_hook)
result = inference_model(model, img_path)
hook.remove()

final_cls_scores = blob_data['cls_scores'][-1, 0]
final_mask_preds = blob_data['mask_preds'][-1, 0]

blob_probabilities = torch.softmax(final_cls_scores, dim=-1).numpy()
blob_masks_binary = (final_mask_preds.numpy() > 0).astype(np.uint8) 

num_blobs = blob_probabilities.shape[0]

# ==========================================
# 3. BUILD DATAFRAME AND MAP
# ==========================================
print("Building DataFrame and visual map...")

orig_img = cv2.imread(img_path)
orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
img_h, img_w, _ = orig_img.shape

mask_h, mask_w = blob_masks_binary.shape[1], blob_masks_binary.shape[2]
scale_y = img_h / mask_h
scale_x = img_w / mask_w

fig, axs = plt.subplots(1, 2, figsize=(20, 10))
axs[0].imshow(orig_img)
axs[0].set_title('1. Original Image', fontsize=16)
axs[0].axis('off')

axs[1].imshow(orig_img)
axs[1].set_title(f'2. Blob Centroid Map (Threshold > {visualization_threshold*100}%)', fontsize=16)
axs[1].axis('off')

# List to hold all our blob data dictionaries
df_records = []

for i in range(num_blobs):
    prob_vec = blob_probabilities[i]
    total_object_prob = 1.0 - prob_vec[no_object_class_idx]
    
    if total_object_prob > visualization_threshold:
        predicted_class_idx = np.argmax(prob_vec[:no_object_class_idx])
        predicted_class_name = class_names[predicted_class_idx]
        
        mask = blob_masks_binary[i]
        coords = np.argwhere(mask) 
        
        if coords.shape[0] > 0:
            centroid_y = int(coords[:, 0].mean() * scale_y)
            centroid_x = int(coords[:, 1].mean() * scale_x)
            
            p_color = point_colors[predicted_class_idx]
            axs[1].scatter(centroid_x, centroid_y, color=p_color, edgecolors='white', s=150, zorder=10)
            axs[1].text(centroid_x + 5, centroid_y + 5, str(i), 
                        color='black', fontsize=10, fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.2'), zorder=11)
            
            # --- CREATE THE DATAFRAME RECORD ---
            record = {
                'blob_id': i,
                'centroid_x': centroid_x,
                'centroid_y': centroid_y,
                'object_confidence': total_object_prob,
                'conditional_prediction': predicted_class_name
            }
            
            # Add probabilities for all actual classes
            for c_idx, c_name in enumerate(class_names):
                record[f'prob_{c_name}'] = prob_vec[c_idx]
                
            # Add the 'No Object' probability
            record['prob_no_object'] = prob_vec[no_object_class_idx]
            
            df_records.append(record)

# ==========================================
# 4. SAVE EVERYTHING
# ==========================================
# Save Image
plt.tight_layout()
plt.savefig(output_img_path, dpi=300, bbox_inches='tight')

# Save DataFrame
df = pd.DataFrame(df_records)
df.to_pickle(output_pkl_path)
df.to_csv(output_csv_path, index=False)

print(f"\n✅ Success!")
print(f"-> Map saved to: {output_img_path}")
print(f"-> DataFrame saved to: {output_pkl_path}")
print(f"-> DataFrame saved to: {output_csv_path}")
