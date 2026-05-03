import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from mmseg.apis import init_model, inference_model

# ==========================================
# 1. CONFIGURATION
# ==========================================
config_file = '/app/config.py'
checkpoint_file = '/app/output/best_mIoU_iter_17000_swin_t.pth' 
img_path = '/app/data/2026-01-19-defect_dataset/images/fretting-bicycle_image_349.jpg'
output_img_path = '/app/blobs/full_pipeline_dashboard.png'

print("Loading model for full pipeline extraction...")
model = init_model(config_file, checkpoint_file, device='cuda:0')

# Dictionary to hold the wiretapped data
pipeline_data = {}

# ==========================================
# 2. THE WIRETAPS (HOOKS)
# ==========================================

# Hook 1: The Backbone (Surveying the Crime Scene)
def backbone_hook(module, args, output):
    # Backbone outputs a tuple of multi-scale features (usually 4 levels)
    if isinstance(output, tuple) or isinstance(output, list):
        pipeline_data['backbone_features'] = [f.detach().cpu() for f in output]
    else:
        pipeline_data['backbone_features'] = [output.detach().cpu()]

# Hook 2: The Pixel Decoder (Drawing the Master Map)
def pixel_decoder_hook(module, args, output):
    # Pixel decoder usually outputs (mask_features, multi_scale_memory)
    # mask_features is the high-res 256-channel map the queries multiply against
    if isinstance(output, tuple):
        pipeline_data['mask_features'] = output[0].detach().cpu()
    else:
        pipeline_data['mask_features'] = output.detach().cpu()

# Hook 3: The Prediction Heads (Writing the Reports)
def head_hook(module, args, output):
    cls_scores, mask_preds = output[0], output[1]
    if isinstance(cls_scores, list): cls_scores = torch.stack(cls_scores)
    if isinstance(mask_preds, list): mask_preds = torch.stack(mask_preds)
    pipeline_data['cls_scores'] = cls_scores.detach().cpu()
    pipeline_data['mask_preds'] = mask_preds.detach().cpu()

# Attach the wiretaps to the specific components of the model
h1 = model.backbone.register_forward_hook(backbone_hook)
h2 = model.decode_head.pixel_decoder.register_forward_hook(pixel_decoder_hook)
h3 = model.decode_head.register_forward_hook(head_hook)

# ==========================================
# 3. RUN INFERENCE & CLEANUP
# ==========================================
print("Running image through the network...")
result = inference_model(model, img_path)

# Remove hooks so they don't consume memory later
h1.remove()
h2.remove()
h3.remove()

# ==========================================
# 4. DATA PROCESSING FOR VISUALIZATION
# ==========================================
print("Processing tensors into visual heatmaps...")

# A helper function to take a 3D tensor (Channels, H, W) and turn it into a 2D heatmap
def tensor_to_heatmap(tensor):
    # Average across all channels to see overall "activation" or "attention"
    heatmap = torch.mean(tensor, dim=0).numpy()
    # Normalize between 0 and 1 for plotting
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-5)
    return heatmap

# Process Step 1: Backbone (Let's take the lowest and highest resolution maps)
bb_features = pipeline_data['backbone_features']
bb_high_res = tensor_to_heatmap(bb_features[0][0]) # Level 0 (Most fine details)
bb_low_res = tensor_to_heatmap(bb_features[-1][0]) # Level 3 (Most abstract/global context)

# Process Step 2: Pixel Decoder
mask_features_heatmap = tensor_to_heatmap(pipeline_data['mask_features'][0])

# Process Step 4: Heads (Get the top 1 confident blob mask)
final_cls_scores = pipeline_data['cls_scores'][-1, 0]
final_mask_preds = pipeline_data['mask_preds'][-1, 0]
probs = torch.softmax(final_cls_scores, dim=-1).numpy()
obj_confidences = 1.0 - probs[:, 9] # Index 9 is "No Object"
top_blob_idx = np.argmax(obj_confidences)
top_blob_mask = (final_mask_preds[top_blob_idx].numpy() > 0).astype(np.float32)

# Process Step 5: Final Post-Processing
final_segmentation = result.pred_sem_seg.data[0].cpu().numpy()

# Load Original Image
orig_img = cv2.imread(img_path)
orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

# ==========================================
# 5. BUILD THE DASHBOARD
# ==========================================
print("Drawing dashboard...")
fig, axs = plt.subplots(2, 3, figsize=(20, 12))
axs = axs.flatten()

def plot_panel(ax, image, title, cmap=None):
    if cmap:
        im = ax.imshow(image, cmap=cmap)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.imshow(image)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')

# 0. Original Image
plot_panel(axs[0], orig_img, "0. Input Image")

# 1A. Backbone (High Res)
plot_panel(axs[1], bb_high_res, "1A. Backbone (Fine Details)\nSwin-T Early Layers", cmap='magma')

# 1B. Backbone (Low Res)
plot_panel(axs[2], bb_low_res, "1B. Backbone (Global Context)\nSwin-T Deep Layers", cmap='magma')

# 2. Pixel Decoder
plot_panel(axs[3], mask_features_heatmap, "2. Pixel Decoder (Master Map)\nFused Multi-Scale Features", cmap='viridis')

# 3 & 4. Top Query Mask
plot_panel(axs[4], top_blob_mask, f"3 & 4. Top Blob Matrix Multiplication\nQuery #{top_blob_idx} x Master Map", cmap='Blues')

# 5. Final Output
plot_panel(axs[5], final_segmentation, "5. Final Post-Processed Mask\n(All 100 Blobs Combined)", cmap='nipy_spectral')

plt.tight_layout()
plt.savefig(output_img_path, dpi=300, bbox_inches='tight')
print(f"✅ Pipeline dashboard saved to: {output_img_path}")