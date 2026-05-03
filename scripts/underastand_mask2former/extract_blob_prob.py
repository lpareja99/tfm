import torch
import numpy as np
from mmseg.apis import init_model, inference_model

# ==========================================
# 1. CONFIGURATION
# ==========================================
config_file = '/app/config.py'
checkpoint_file = '/app/output/best_mIoU_iter_17000_swin_t.pth'
img_path = '/app/data/2026-01-19-defect_dataset/images/fretting-bicycle_image_349.jpg'

# Initialize the model
model = init_model(config_file, checkpoint_file, device='cuda:0')

# ==========================================
# 2. THE HOOK (INTERCEPTING THE BLOBS)
# ==========================================
blob_data = {}

def mask2former_hook(module, args, output):
    cls_scores = output[0]
    mask_preds = output[1]
    
    # If the output is a list (one tensor per transformer layer), stack them
    if isinstance(cls_scores, list):
        cls_scores = torch.stack(cls_scores)
    if isinstance(mask_preds, list):
        mask_preds = torch.stack(mask_preds)
        
    # Now it is guaranteed to be a tensor, so we can safely detach
    blob_data['cls_scores'] = cls_scores.detach().cpu()
    blob_data['mask_preds'] = mask_preds.detach().cpu()

# Attach the hook directly to the model's decode head
hook = model.decode_head.register_forward_hook(mask2former_hook)

# ==========================================
# 3. RUN INFERENCE
# ==========================================
print(f"Running image through Mask2Former...")
# This automatically triggers the hook and populates our blob_data dictionary
result = inference_model(model, img_path)

# Important: Remove the hook so it doesn't cause memory leaks later
hook.remove()

# ==========================================
# 4. EXTRACT PROBABILITIES PER BLOB
# ==========================================
# - We take the last transformer layer: index [-1]
# - We take the first image in the batch: index [0]
final_cls_scores = blob_data['cls_scores'][-1, 0]
final_mask_preds = blob_data['mask_preds'][-1, 0]

# Apply Softmax along the class dimension to get percentages (0 to 1)
# Note: In CrossEntropy setups, Mask2Former usually adds 1 extra class at the end 
# representing "background / no object" for queries that didn't find anything.
blob_probabilities = torch.softmax(final_cls_scores, dim=-1).numpy()
blob_masks = final_mask_preds.numpy()

num_blobs = blob_probabilities.shape[0]
num_classes_output = blob_probabilities.shape[1]

print(f"\n✅ Success! Intercepted {num_blobs} individual blobs/queries.")
print(f"Probabilities shape: {blob_probabilities.shape} -> (blobs, classes)")
print(f"Masks shape: {blob_masks.shape} -> (blobs, height, width)")

# ==========================================
# 5. DEMO THE DATA
# ==========================================
print("\n--- Example: Analysis for Blob #0 ---")
print("Percentage of being in each class:")
for class_idx in range(num_classes_output):
    prob = blob_probabilities[0, class_idx]
    
    # Try to map to your class names if it falls within the 0-8 range
    if class_idx < 9:
        class_names = ["bg", "cracks", "cracks_alligator", "cracks_severe", 
                       "edge_cracks", "fretting", "pothole", "manhole", "pole_shadow"]
        name = class_names[class_idx]
    else:
        name = "Mask2Former 'No Object' Class"
        
    print(f"  {name}: {prob * 100:>6.2f}%")

# You can now save these specific blob metrics if you want
# np.save('blob_probabilities.npy', blob_probabilities)
# np.save('blob_masks.npy', blob_masks)