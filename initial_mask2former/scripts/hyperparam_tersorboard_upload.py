import os
from mmengine.config import Config
from torch.utils.tensorboard import SummaryWriter

# 1. Setup paths
work_dir = './work_dirs/subset_500_flowity_bg'
config_path = os.path.join(work_dir, 'subset_500_flowity_bg.py') 
log_dir = os.path.join(work_dir, 'results/vis_data')

if not os.path.exists(config_path):
    print(f"❌ Error: Could not find config at {config_path}")
    exit()

# 2. Load the config automatically
cfg = Config.fromfile(config_path)


def get_pipeline_step(pipeline, step_type):
    for step in pipeline:
        if step.get('type') == step_type:
            return step
    return {}

# 3. Extract Hyperparameters automatically
hparams = {
    # --- Classes---
    'num_classes': cfg.model.decode_head.num_classes,
    'reduce_zero_label': cfg.model.decode_head.get('reduce_zero_label', False),
    
    # --- Optimization ---
    'lr': cfg.optim_wrapper.optimizer.lr,
    'batch_size': cfg.train_dataloader.batch_size,
    'total_iters': cfg.train_cfg.max_iters,
    'weight_decay': cfg.optim_wrapper.optimizer.weight_decay,
    'optimizer': cfg.optim_wrapper.optimizer.type,
    'poly_power': cfg.param_scheduler[0].get('power', 0.9),
    
    # --- Architecture ---
    'backbone': cfg.model.backbone.type,
    'swin_depths': str(cfg.model.backbone.get('depths', [])),
    'num_classes': cfg.model.decode_head.num_classes,
    'num_queries': cfg.model.decode_head.get('num_queries', 'N/A'),
    'feat_channels': cfg.model.decode_head.feat_channels,
    'loss_weight_ce': cfg.model.decode_head.loss_cls.loss_weight,
    'loss_weight_dice': cfg.model.decode_head.loss_dice.loss_weight,
    'loss_weight_mask': cfg.model.decode_head.loss_mask.loss_weight,
    
    # --- Resolutions & Data ---
    # We look for the 'RandomResize' step to get the input resolution
    'input_scale': str(get_pipeline_step(cfg.train_pipeline, 'RandomResize').get('scale', 'N/A')),
    'crop_size': str(get_pipeline_step(cfg.train_pipeline, 'RandomCrop').get('crop_size', 'N/A')),
    'mean': str(cfg.model.data_preprocessor.get('mean', [])),
    
    # --- Augmentation Meta ---
    'cat_max_ratio': get_pipeline_step(cfg.train_pipeline, 'RandomCrop').get('cat_max_ratio', 'N/A'),
    'ratio_range': str(get_pipeline_step(cfg.train_pipeline, 'RandomResize').get('ratio_range', 'N/A')),
    'norm_mean': str(cfg.model.data_preprocessor.get('mean', 'N/A'))
}

# 4. Define target metrics to link in TensorBoard
metrics = {
    'mIoU': 0.0,
    'mDice': 0.0,
    'loss': 0.0
}

# 5. Inject into TensorBoard
writer = SummaryWriter(log_dir)
writer.add_hparams(hparams, metrics)
writer.close()

print(f"✅ Successfully extracted and synced hparams from {config_path}")
print(f"Extracted: LR={hparams['lr']}, Batch={hparams['batch_size']}, Iters={hparams['total_iters']}")