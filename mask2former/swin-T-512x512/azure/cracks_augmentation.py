data_root = None

_base_ = ['mmseg::mask2former/mask2former_swin-t_8xb2-160k_ade20k-512x512.py']

#class_names = ("bg", "cracks", "cracks_alligator", "cracks_severe")
resume = True

""" palette = [
    [0, 0, 0],       # bg - Black
    [250, 50, 83],   # cracks - Red/Pink
    [36, 179, 83],   # cracks_alligator - Green
    [102, 204, 255]  # cracks_severe - Light Green
] """

""" class_names = ("bg", "cracks", "cracks_alligator", "cracks_severe", "edge_cracks", "fretting", "pothole", "manhole", "pole_shadow")
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
] """

class_names = (
    "bg", "cracks", "cracks_alligator", "cracks_severe", 
    "edge_breaks", "fretting", "pothole", "manhole", 
    "patched", "bad_joint", "joint", "large_repair", 
    "loose_stones", "pole_shadow", "sill", "tyre_mark", 
    "edge_grass"
)

palette = [
    [0, 0, 0],         # 0: bg
    [250, 50, 83],     # 1: cracks
    [36, 179, 83],     # 2: cracks_alligator
    [102, 255, 102],   # 3: cracks_severe
    [255, 0, 255],     # 4: edge_breaks
    [204, 153, 51],    # 5: fretting
    [115, 51, 128],    # 6: pothole
    [34, 62, 209],     # 7: manhole
    [63, 63, 63],      # 8: patched
    [224, 68, 45],     # 9: bad_joint
    [255, 153, 51],    # 10: joint
    [255, 255, 51],    # 11: large_repair
    [51, 255, 255],    # 12: loose_stones
    [172, 84, 109],    # 13: pole_shadow
    [36, 223, 0],      # 14: sill
    [170, 68, 22],     # 15: tyre_mark
    [213, 164, 25]     # 16: edge_grass
]

metainfo = dict(
    classes=class_names,
    palette=palette
)
   
log_level = 'INFO'
work_dir = './work_dirs/cracks_augmentation'

# Iteration Logic
dataset_type = 'BaseSegDataset'

img_num = 5000
batch_size = 8
num_workers = 2
val_interval = 1000
log_interval = 100
num_classes = len(class_names)


iter_per_epoch = img_num / batch_size
epochs = 30
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
            class_weight=[0.1] + [1.0] * (num_classes - 1) + [0.1] #class adjustment, less weight on background 
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
        patience=4,          # Number of validations to wait
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
        max_keep_ckpts=3, 
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

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False), # Crucial fix
    dict(type='RandomResize', scale=(2048, 512), ratio_range=(0.5, 2.0), keep_ratio=True),
    
    # Handle grainy images/motion blur
    #dict(type='RandomChoice', transforms=[
    #    [dict(type='GaussianBlur', sigma=(0.1, 2.0))],
    #    [dict(type='GaussianNoise', sigma=(0, 0.05))],
    #    [dict(type='Identity')]
    #]),
    
    # Handle lighting/contrast variations
    dict(type='PhotoMetricDistortion'),
    
    dict(type='mmcv.RandomGrayscale', prob=0.1, keep_channels=True),
    
    # Force focus on structure over color
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction=['horizontal', 'vertical']),
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