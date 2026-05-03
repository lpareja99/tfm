import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from mmseg.apis import init_model, inference_model

# ==========================================
# 1. MASTER CONFIGURATION
# ==========================================
config_file = '/app/config.py'
checkpoint_file = '/app/output/best_mIoU_iter_17000_swin_t.pth' # UPDATE IF NEEDED
img_path = '/app/data/2026-01-19-defect_dataset/images/fretting-bicycle_image_349.jpg'# UPDATE IF NEEDED
output_dir = '/app/blobs'

os.makedirs(output_dir, exist_ok=True)

class_names = np.array(["bg", "cracks", "cracks_alligator", "cracks_severe", 
                        "edge_cracks", "fretting", "pothole", "manhole", "pole_shadow"])
no_obj_idx = 9

print(f"\n[SYSTEM] Initializing Master Storyteller for: {img_path.split('/')[-1]}")
model = init_model(config_file, checkpoint_file, device='cuda:0')

# ==========================================
# 2. THE GRAND WIRETAP (HOOKS)
# ==========================================
story_data = {}

def backbone_hook(module, args, output):
    if isinstance(output, tuple) or isinstance(output, list):
        story_data['backbone'] = [f.detach().cpu() for f in output]
    else:
        story_data['backbone'] = [output.detach().cpu()]

def pixel_decoder_hook(module, args, output):
    if isinstance(output, tuple):
        story_data['master_map'] = output[0].detach().cpu()
    else:
        story_data['master_map'] = output.detach().cpu()

def head_hook(module, args, output):
    cls_scores, mask_preds = output[0], output[1]
    if isinstance(cls_scores, list): cls_scores = torch.stack(cls_scores)
    if isinstance(mask_preds, list): mask_preds = torch.stack(mask_preds)
    story_data['cls_scores'] = cls_scores.detach().cpu()
    story_data['mask_preds'] = mask_preds.detach().cpu()

h1 = model.backbone.register_forward_hook(backbone_hook)
h2 = model.decode_head.pixel_decoder.register_forward_hook(pixel_decoder_hook)
h3 = model.decode_head.register_forward_hook(head_hook)

print("[SYSTEM] Running image through the entire Mask2Former assembly line...")
result = inference_model(model, img_path)

h1.remove()
h2.remove()
h3.remove()

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def to_heatmap(tensor):
    hm = torch.mean(tensor, dim=0).numpy()
    return (hm - np.min(hm)) / (np.max(hm) - np.min(hm) + 1e-5)

orig_img = cv2.imread(img_path)
orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
img_h, img_w = orig_img.shape[:2]

# ==========================================
# ACT 1: FEATURE EXTRACTION (THE CARTOGRAPHERS)
# ==========================================
print("\n[ACT 1] Generating 'Translating the Real World into Math' Dashboard...")

fig1, axs1 = plt.subplots(1, 4, figsize=(24, 6))
fig1.suptitle("ACT 1: Translating the Real World into Math (Backbone & Pixel Decoder)", fontsize=20, fontweight='bold')

bb_high = to_heatmap(story_data['backbone'][0][0])  # Level 0 (Edges/Textures)
bb_low = to_heatmap(story_data['backbone'][-1][0])  # Level 3 (Global Context/Blobs)
master_map = to_heatmap(story_data['master_map'][0]) # Fused 256-D Map

axs1[0].imshow(orig_img)
axs1[0].set_title("1. Original Image", fontsize=14)
axs1[0].axis('off')

axs1[1].imshow(bb_high, cmap='magma')
axs1[1].set_title("2. Swin-T Early Layer\n(Extracting fine details & edges)", fontsize=14)
axs1[1].axis('off')

axs1[2].imshow(bb_low, cmap='magma')
axs1[2].set_title("3. Swin-T Deep Layer\n(Extracting global context & abstract shapes)", fontsize=14)
axs1[2].axis('off')

axs1[3].imshow(master_map, cmap='viridis')
axs1[3].set_title("4. Pixel Decoder Output\n(The Fused 256-D Master Feature Map)", fontsize=14)
axs1[3].axis('off')

plt.tight_layout()
fig1.savefig(os.path.join(output_dir, 'Act_1_Feature_Extraction.png'), dpi=300)
plt.close(fig1)

# ==========================================
# ACT 2: THE 100 DETECTIVES (PROPOSALS)
# ==========================================
print("[ACT 2] Generating 'The 100 Detectives' Dashboard...")

final_cls_scores = story_data['cls_scores'][-1, 0]
final_mask_preds = story_data['mask_preds'][-1, 0]
probs = torch.softmax(final_cls_scores, dim=-1).numpy()
masks_sigmoid = torch.sigmoid(final_mask_preds).numpy()

# Find the Top 4 most confident blobs
obj_confidences = 1.0 - probs[:, no_obj_idx]
top_4_indices = np.argsort(obj_confidences)[-4:][::-1]

fig2, axs2 = plt.subplots(2, 4, figsize=(24, 12))
fig2.suptitle("ACT 2: The Detectives Present Their Findings (Top 4 Queries)", fontsize=20, fontweight='bold')

for i, blob_idx in enumerate(top_4_indices):
    prob_vec = probs[blob_idx]
    pred_idx = np.argmax(prob_vec[:no_obj_idx])
    pred_name = class_names[pred_idx]
    conf = obj_confidences[blob_idx] * 100
    
    # Resize mask to original image size
    mask_resized = cv2.resize(masks_sigmoid[blob_idx], (img_w, img_h))
    overlay = np.zeros((img_h, img_w, 4), dtype=np.float32)
    overlay[mask_resized > 0.5] = [0.0, 1.0, 0.0, 0.6] # Green overlay
    
    # Top Row: The Mask Proposal
    axs2[0, i].imshow(orig_img)
    axs2[0, i].imshow(overlay)
    axs2[0, i].set_title(f"Blob #{blob_idx} Proposal\nConfidence: {conf:.1f}%", fontsize=14, fontweight='bold')
    axs2[0, i].axis('off')
    
    # Bottom Row: The Classification Math
    top_3_cls_idx = np.argsort(prob_vec[:no_obj_idx])[-3:][::-1] # Exclude 'no object' for clarity
    top_3_names = class_names[top_3_cls_idx]
    top_3_vals = prob_vec[top_3_cls_idx]
    
    axs2[1, i].barh(top_3_names[::-1], top_3_vals[::-1], color='steelblue')
    axs2[1, i].set_title(f"Blob #{blob_idx} Identity Crisis\nWinner: {pred_name.upper()}", fontsize=14)
    axs2[1, i].set_xlim(0, 1.0)

plt.tight_layout()
fig2.savefig(os.path.join(output_dir, 'Act_2_The_Detectives.png'), dpi=300)
plt.close(fig2)

# ==========================================
# ACT 3: THE FINAL VOTE & DEMOCRACY
# ==========================================
print("[ACT 3] Generating 'The Final Democratic Vote' Dashboard...")

pixel_probs = torch.softmax(result.seg_logits.data, dim=0).cpu().numpy()
final_seg_map = result.pred_sem_seg.data[0].cpu().numpy()

# Calculate Margin of Victory (Uncertainty)
sorted_probs = np.sort(pixel_probs, axis=0)
margin = sorted_probs[-1, :, :] - sorted_probs[-2, :, :]
winning_class = np.argmax(pixel_probs, axis=0)

# Create an Uncertainty Heatmap (Low margin = High Uncertainty = Red)
# We only care about uncertainty where the model predicted a defect (not background)
uncertainty_map = np.zeros_like(margin)
defect_mask = (winning_class != 0)
uncertainty_map[defect_mask] = 1.0 - margin[defect_mask] 

fig3, axs3 = plt.subplots(1, 3, figsize=(24, 8))
fig3.suptitle("ACT 3: The Final Democratic Vote (Pixel-by-Pixel Battle)", fontsize=20, fontweight='bold')

# 1. All Blob Boundaries Overlaid
axs3[0].imshow(orig_img)
for blob_idx in top_4_indices:
    mask_resized = cv2.resize(masks_sigmoid[blob_idx], (img_w, img_h))
    axs3[0].contour(mask_resized, levels=[0.5], colors=['red'], linewidths=1.5, alpha=0.6)
axs3[0].set_title("1. All Detectives' Claims Overlapping\n(Notice the border clashes)", fontsize=14)
axs3[0].axis('off')

# 2. The Battleground Heatmap
im = axs3[1].imshow(uncertainty_map, cmap='hot', vmin=0, vmax=1.0)
axs3[1].set_title("2. Pixel Battlegrounds (Uncertainty)\n(Bright colors = close votes between defects)", fontsize=14)
fig3.colorbar(im, ax=axs3[1], shrink=0.7)
axs3[1].axis('off')

# 3. The Final Output
im2 = axs3[2].imshow(final_seg_map, cmap='nipy_spectral', vmin=0, vmax=8)
axs3[2].set_title("3. The Final Output Mask\n(After 'Argmax' resolves the votes)", fontsize=14)
fig3.colorbar(im2, ax=axs3[2], ticks=range(9), shrink=0.7)
axs3[2].axis('off')

plt.tight_layout()
fig3.savefig(os.path.join(output_dir, 'Act_3_Final_Verdict.png'), dpi=300)
plt.close(fig3)

print("\n[SYSTEM] Master Storyteller Complete!")
print("Please check your results folder for:")
print(" -> Act_1_Feature_Extraction.png")
print(" -> Act_2_The_Detectives.png")
print(" -> Act_3_Final_Verdict.png")