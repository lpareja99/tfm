import sys
import torch
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from mmseg.apis import init_model

# Usage: python model_complexity_measurement.py <config.py> [checkpoint.pth]
# checkpoint is optional (parameter/GFLOP counts do not need trained weights).
config_file = sys.argv[1] if len(sys.argv) > 1 else 'config.py'
checkpoint_file = sys.argv[2] if len(sys.argv) > 2 else None
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print(f"--- Analyzing: {config_file} (checkpoint: {checkpoint_file or 'none'}, device: {device}) ---")

model = init_model(config_file, checkpoint_file, device=device)
model.eval()

# 1. Total Parameters (Full Model)
# This works regardless of the forward pass requirements
total_params = sum(p.numel() for p in model.parameters())

# 2. GFLOPs (Isolating the Backbone)
# We analyze the backbone because Mask2FormerHead forward is too complex for tracers
dummy_input = torch.randn((1, 3, 512, 512)).to(device)
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