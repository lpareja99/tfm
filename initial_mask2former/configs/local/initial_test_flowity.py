import os
from tensorboardX.summary import SummaryWriter

_base_ = ['mmseg::mask2former/mask2former_swin-t_8xb2-160k_ade20k-512x512.py']
log_level = 'INFO'
work_dir = './work_dirs/initial_test_flowity'

# Iteration Logic
dataset_type = 'BaseSegDataset'
data_root = 'data/flowity_test/some_defects'

# Calculate the number of iterations based on your dataset and desired epochs
train_dir = f'{data_root}/images/training'
num_imgs = len([f for f in os.listdir(train_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

target_epochs = 100 
val_per_epoch = 10 # How many times per training session to validate
batch_size = 2
total_iters = (num_imgs // batch_size) * target_epochs
val_interval = max(1, total_iters // val_per_epoch)

print(f"---> Training for {target_epochs} epochs ({total_iters} total iterations).")


# 1. Model Config
model = dict(
    decode_head=dict(
        num_classes=17, # Change from 150 to 17
        out_channels=17, # Change from 256 to 17
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * 18) # 17 classes + bg
    )
)


# 3. Early Stopping and Hooks
custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='mIoU',      # Metric to monitor
        rule='greater',      # Stop if mIoU stops increasing
        min_delta=0.005,     # Minimum change to count as an improvement
        patience=3,          # Number of validations to wait
    )
]

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', 
        by_epoch=False, 
        interval=val_interval, 
        max_keep_ckpts=3, 
        save_best='mIoU',
        out_dir=f'{work_dir}/checkpoints',
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(
        type='SegVisualizationHook', 
        draw=False,
        interval=1)
    
)



layout = {
    "Performance": {
        "Accuracy": ["Multiline", ["aAcc", "mAcc"]],
        "Loss": ["Multiline", ["loss", "loss_ce", "loss_mask"]],
    },
    "Validation": {
        "mIoU": ["Multiline", ["mIoU", "mDice"]],
    }
}

vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend',
                    save_dir=f'{work_dir}/results/vis_data', 
                    init_cfg=dict(layout=layout))]

visualizer = dict(
    type='SegLocalVisualizer', 
    vis_backends=vis_backends, 
    save_dir=f'{work_dir}/results',
    name='visualizer',
    alpha=0.6
)

# Ensure the evaluator is present so the hook has data to monitor
val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU', 'mDice', 'mFscore'])


test_evaluator = val_evaluator

# Metadata
class_names = ("bg", "cracks", "cracks_alligator", "cracks_severe", "edge_breaks", 
               "fretting", "pothole", "manhole", "patched", "bad_joint", "joint", 
               "large_repair", "loose_stones", "pole_shadow", "sill", "tyre_mark", "edge_grass")

palette = [[0, 0, 0], [250, 50, 83], [36, 179, 83], [102, 255, 102], [255, 0, 255],
           [204, 153, 51], [115, 51, 128], [34, 62, 209], [63, 63, 63], [224, 68, 45],
           [255, 153, 51], [255, 255, 51], [51, 255, 255], [172, 84, 109], [36, 223, 0],
           [170, 68, 22], [213, 164, 25]]

metainfo = dict(
    classes=class_names,
    palette=palette
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False), # Crucial fix
    dict(type='RandomResize', scale=(2048, 512), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False), # Crucial fix
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=batch_size, # Safety for your 4070
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images/training', seg_map_path='labels/training'),
        metainfo=metainfo,
        pipeline=train_pipeline,
        reduce_zero_label=False))

val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images/validation', seg_map_path='labels/validation'),
        metainfo=metainfo,
        pipeline=test_pipeline,
        reduce_zero_label=False))

test_dataloader = val_dataloader


# Running Settings
#log_file = f'{work_dir}/training_progress.log'
work_dir = work_dir
train_cfg = dict(type='IterBasedTrainLoop', max_iters=total_iters, val_interval=val_interval)
param_scheduler = [
    dict(type='PolyLR', begin=0, end=total_iters, power=0.9, by_epoch=False)
]