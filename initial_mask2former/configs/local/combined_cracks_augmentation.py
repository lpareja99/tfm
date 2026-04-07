_base_ = ['mmseg::mask2former/mask2former_swin-t_8xb2-160k_ade20k-512x512.py']

class_names = ("bg", "cracks")

palette = [[0, 0, 0], [250, 50, 83]]

metainfo = dict(
    classes=class_names,
    palette=palette
)
   
log_level = 'INFO'
work_dir = './work_dirs/combined_cracks_augmentation'

# Iteration Logic
dataset_type = 'BaseSegDataset'
data_root = 'data/combine_crack'

batch_size = 2
max_iterations = 3000 # 750 iters * 5 epochs
val_interval = 750
num_classes = 2

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
            # [Background, Cracks, No-Object]
            class_weight=[1.0, 2.0, 1.0]
        )
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
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
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
        interval=10)
    
)

vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend',
                    save_dir=f'{work_dir}/results/vis_data')
                ]

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
    
    #dict(type='RandomGrayscale', prob=0.1),
    dict(type='mmcv.RandomGrayscale', prob=0.1, keep_channels=True),
    
    # Force focus on structure over color
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    # Rotate by up to 45 degrees to handle diagonal cracks
    dict(type='RandomRotate', prob=0.5, degree=45, pad_val=0, seg_pad_val=0),
    dict(type='RandomFlip', prob=0.5, direction=['horizontal', 'vertical']),
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

test_dataloader = val_dataloader

# Running Settings
work_dir = work_dir

train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',  # Changed from EpochBasedTrainLoop
    max_iters= max_iterations,             # 750 iters * 5 epochs
    val_interval= val_interval           # Validate exactly once per "epoch"
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='PolyLR', begin=0, end=max_iterations, power=0.9, by_epoch=False)
]