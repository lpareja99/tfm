import mmengine
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mmseg.apis import init_model, inference_model
from mmseg.registry import DATASETS
from sklearn.metrics import confusion_matrix
import os

# --- SETTINGS ---
config_file = 'config_tiny.py'
checkpoint_file = 'output/best_mIoU_iter_28000_flash.pth'
class_names = ["bg", "cracks", "cracks_alligator", "cracks_severe", "edge_cracks", "fretting", "pothole", "manhole", "pole_shadow"]

print("--- Initializing Confusion Matrix System ---")
model = init_model(config_file, checkpoint_file, device='cuda:0')
cfg = mmengine.Config.fromfile(config_file)
dataset = DATASETS.build(cfg.test_dataloader.dataset)

# 9x9 matrix for our 9 classes
total_cm = np.zeros((9, 9), dtype=np.int64)

print(f"--- Processing {len(dataset)} images ---")
with torch.no_grad():
    for i in range(len(dataset)):
        # 1. Get the Ground Truth from the dataset object
        data_batch = dataset[i]
        # gt_sem_seg is inside the data_samples of the dataset item
        gt = data_batch['data_samples'].gt_sem_seg.data[0].cpu().numpy().flatten()
        
        # 2. Get the Prediction from the model
        img_path = data_batch['data_samples'].img_path
        result = inference_model(model, img_path)
        pred = result.pred_sem_seg.data[0].cpu().numpy().flatten()
        
        # 3. Filter out ignore_index (255) and keep only classes 0-8
        mask = (gt >= 0) & (gt < 9)
        
        if np.any(mask):
            # Calculate matrix for this image and add to total
            total_cm += confusion_matrix(gt[mask], pred[mask], labels=range(9))
        
        if (i + 1) % 50 == 0:
            print(f"Progress: {i+1}/{len(dataset)} images processed...")

# --- PLOTTING ---
print("--- Generating Normalized Heatmap ---")
# Normalize by row (Recall) to see percentages of each class
row_sums = total_cm.sum(axis=1, keepdims=True)
cm_normalized = np.divide(total_cm.astype(float), row_sums, 
                          out=np.zeros_like(total_cm, dtype=float), 
                          where=row_sums != 0)

plt.figure(figsize=(12, 10))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix: Flash-InternImage-T')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')

if not os.path.exists('output'): os.makedirs('output')
plt.savefig('output/confusion_matrix_flash_internimage_t.png', bbox_inches='tight')
print("--- FINISHED! Your matrix is at: output/confusion_matrix_flash_internimage_t.png ---")