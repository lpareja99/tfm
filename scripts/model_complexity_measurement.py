import torch
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from mmseg.apis import init_model

config_file = 'config.py'
checkpoint_file = 'output/best_mIoU_iter_17000_swin_t.pth'

print(f"--- Analyzing: {checkpoint_file} ---")

model = init_model(config_file, checkpoint_file, device='cuda:0')
model.eval()

# 1. Total Parameters (Full Model)
# This works regardless of the forward pass requirements
total_params = sum(p.numel() for p in model.parameters())

# 2. GFLOPs (Isolating the Backbone)
# We analyze the backbone because Mask2FormerHead forward is too complex for tracers
dummy_input = torch.randn((1, 3, 512, 512)).cuda()
flops = FlopCountAnalysis(model.backbone, dummy_input)
backbone_flops = flops.total()

print("\n" + "="*30)
print("COMPLEXITY RESULTS")
print("="*30)
print(f"Full Model Parameters: {total_params / 1e6:.2f} M")
print(f"Backbone GFLOPs: {backbone_flops / 1e9:.2f} G")
print("="*30)

# This table will still show you the breakdown per component
print("\nDetailed Parameter Breakdown:")
print(parameter_count_table(model, max_depth=2))