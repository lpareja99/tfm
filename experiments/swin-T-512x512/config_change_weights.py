# ===========================================================================
# EXPERIMENT 1A: Swin-T-512x512 - Crack Only CONFIG
#
# Note: This configuration is set for Azure, the local configuration
#       is changed dynamically on the Makefile
# ===========================================================================

data_root = "/app/data/2026-01-19-defect_dataset"
work_dir = "default"
resume = True

_base_ = ['mmseg::mask2former/mask2former_swin-t_8xb2-160k_ade20k-512x512.py']

class_names = ("bg", "cracks", "cracks_alligator", "cracks_severe", "edge_cracks", "fretting", "pothole", "manhole", "pole_shadow")
palette = [
    [0, 0, 0],       # bg - Black
    [250, 50, 83],   # cracks - Red/Pink
    [36, 179, 83],   # cracks_alligator - Green
    [102, 204, 255], # cracks_severe - Light Green
    [255, 165, 0],   # edge_cracks - Orange
    [128, 128, 128], # fretting - Gray
    [255, 255, 0],   # pothole - Yellow
    [0, 255, 255],   # manhole - Cyan
    [255, 0, 255]    # pole_shadow - Magenta
]

metainfo = dict(
    classes=class_names,
    palette=palette
)
   
log_level = 'INFO'
dataset_type = 'BaseSegDataset'

num_classes = len(class_names)
img_num = 5800
batch_size = 8
num_workers = 2
val_interval = 1000
log_interval = 100

iter_per_epoch = img_num / batch_size
epochs = 80
max_iterations = int(iter_per_epoch * epochs)
crop_size = (512, 512)

print(f"---> Training for {max_iterations} iterations.")

# 1. Model Config
model = dict(
    decode_head=dict(
        num_classes=num_classes, 
        out_channels=num_classes,
        ignore_index=255,
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0, 10.5176, 7.1154, 22.0909, 51.8898, 7.8277, 33.5274, 51.0165, 68.9769, 1.0]
        )
    )
)

# 3. Early Stopping and Hooks
custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='mIoU',      # Metric to monitor
        rule='greater',      # Stop if mIoU stops increasing
        min_delta=0.003,     # Minimum change to count as an improvement
        patience=5,          # Number of validations to wait
    )
]

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=log_interval, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', 
        by_epoch=False, 
        interval=val_interval, 
        max_keep_ckpts=5, 
        save_best='mIoU',
        out_dir=f'{work_dir}/checkpoints',
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(
        type='SegVisualizationHook', 
        draw=True,
        interval=10)
    
)

visualizer = dict(
    type='SegLocalVisualizer', 
    vis_backends=[dict(type='LocalVisBackend')], 
    save_dir=f'{work_dir}/results',
    name='visualizer',
    alpha=0.6
)

# Ensure the evaluator is present so the hook has data to monitor
val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU', 'mDice', 'mFscore'],
    output_dir=f'{work_dir}/eval_results'
)

test_evaluator = val_evaluator

albu_train_transforms = [
    dict(
        type='OneOf',
        transforms=[
            # Blur limit defines kernel sizes (must be odd). 
            dict(type='GaussianBlur', blur_limit=(1, 3), p=1.0),
            # Variance limit controls the severity of the noise.
            dict(type='GaussNoise', var_limit=(5.0, 30.0), p=1.0),
        ],
        p=0.3  # 30% chance to apply either Blur or Noise. 70% chance to do nothing (Identity).
    )
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False), # Crucial fix
    dict(type='RandomResize', scale=(2048, 512), ratio_range=(0.5, 2.0), keep_ratio=True),
    
    # Investigate if I should add this or keep what I currently have 
    
    # dict(type='RandomResize', 
    #      scale=(1280, 720), 
    #      ratio_range=(0.5, 2.0), # Wider range to simulate GoPro vs Phone
    #      keep_ratio=True), # MANDATORY to prevent defect distortion
    
    # # Pad to the largest possible size your hardware produces (e.g., 720p)
    # # This makes the "different sizes" uniform for the GPU tensors
    # dict(type='Pad', size=(1280, 720), pad_val=dict(img=(0, 0, 0), mask=0)),
    
    
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),

    dict(
        type='Albu',
        transforms=albu_train_transforms,
        keymap=dict(img='image', gt_seg_map='mask'),
        update_pad_shape=False,
    ),
    
    dict(type='PhotoMetricDistortion'),
    dict(type='mmcv.RandomGrayscale', prob=0.1, keep_channels=True),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='splits/train.txt',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        data_prefix=dict(img_path='images', seg_map_path='labels'),
        metainfo=metainfo,
        pipeline=train_pipeline,
        reduce_zero_label=False))

val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='splits/val.txt',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        data_prefix=dict(img_path='images', seg_map_path='labels'),
        metainfo=metainfo,
        pipeline=test_pipeline,
        reduce_zero_label=False))

test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='splits/test.txt',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        data_prefix=dict(img_path='images', seg_map_path='labels'),
        metainfo=metainfo,
        pipeline=test_pipeline,
        reduce_zero_label=False))


# Running Settings
work_dir = work_dir

train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',  
    max_iters= max_iterations,             
    val_interval= val_interval
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='PolyLR', begin=0, end=max_iterations, power=0.9, by_epoch=False)
]