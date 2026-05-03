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
output_path = '/app/blobs/Act1_Deep_Dive.png'

print(f"Executing Act 1 Deep Dive on: {img_path.split('/')[-1]}")
model = init_model(config_file, checkpoint_file, device='cuda:0')

# ==========================================
# 2. THE WIRETAPS
# ==========================================
act1_data = {}

def backbone_hook(module, args, output):
    # Capture all 4 layers of the Feature Pyramid
    if isinstance(output, (tuple, list)):
        act1_data['backbone'] = [f.detach().cpu() for f in output]

def pixel_decoder_hook(module, args, output):
    # Capture the final Master Map (mask_feature) AND the Transformer-enhanced layers
    if isinstance(output, tuple):
        act1_data['master_map'] = output[0].detach().cpu()
        act1_data['enhanced_layers'] = [f.detach().cpu() for f in output[1]]

h1 = model.backbone.register_forward_hook(backbone_hook)
h2 = model.decode_head.pixel_decoder.register_forward_hook(pixel_decoder_hook)

result = inference_model(model, img_path)

h1.remove()
h2.remove()

# ==========================================
# 3. VISUALIZATION HELPERS
# ==========================================
def to_heatmap(tensor):
    hm = torch.mean(tensor, dim=0).numpy()
    return (hm - np.min(hm)) / (np.max(hm) - np.min(hm) + 1e-5)

orig_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

# Extract backbone layers (Raw evidence)
bb_layer1 = to_heatmap(act1_data['backbone'][0][0]) # 1/4 scale
bb_layer2 = to_heatmap(act1_data['backbone'][1][0]) # 1/8 scale
bb_layer3 = to_heatmap(act1_data['backbone'][2][0]) # 1/16 scale
bb_layer4 = to_heatmap(act1_data['backbone'][3][0]) # 1/32 scale

# Extract the Missing Link & Final Output
enhanced_deep = to_heatmap(act1_data['enhanced_layers'][-1][0]) # The Transformer's view of Layer 4
master_map = to_heatmap(act1_data['master_map'][0])

# ==========================================
# 4. DRAWING THE DASHBOARD
# ==========================================
fig, axs = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle("ACT 1: The Backbone and The Missing Link", fontsize=22, fontweight='bold')
axs = axs.flatten()

# Row 1: The Raw Backbone Output
axs[0].imshow(bb_layer1, cmap='magma')
axs[0].set_title("1. Backbone Layer 1 (1/4 Scale)\nCrisp Edges & High-Frequency Noise", fontsize=14)

axs[1].imshow(bb_layer3, cmap='magma')
axs[1].set_title("2. Backbone Layer 3 (1/16 Scale)\nFinding Textures and Regions", fontsize=14)

axs[2].imshow(bb_layer4, cmap='magma')
axs[2].set_title("3. Backbone Layer 4 (1/32 Scale)\nTiny Resolution, Global Context", fontsize=14)

# Row 2: The Pixel Decoder's Work
axs[3].imshow(orig_img)
axs[3].set_title("4. Original Image\n(For Reference)", fontsize=14)

axs[4].imshow(enhanced_deep, cmap='plasma')
axs[4].set_title("5. The Missing Link (Transformer Encoder)\nLayer 4 after 'talking' to other layers", fontsize=14)

axs[5].imshow(master_map, cmap='viridis')
axs[5].set_title("6. The Final Master Map\nUpsampled & Fused 256-D Feature Space", fontsize=14)

for ax in axs:
    ax.axis('off')

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved Act 1 Deep Dive to: {output_path}")