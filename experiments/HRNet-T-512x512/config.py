_base_ = ['mmseg::mask2former/mask2former_r50_8xb2-160k_ade20k-512x512.py']

data_root =  "data/2026-01-19-defect_dataset/"
label_dir = "labels_basic_defects_relabel"
log_level = 'INFO'
work_dir = None
dataset_type = 'BaseSegDataset'

resume = True

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
    type='EncoderDecoder',
    backbone=dict(
        type='HRNet',
        _delete_=True,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        extra=dict(
            stage1=dict(num_modules=1, num_branches=1, block='BOTTLENECK', num_blocks=(4,), num_channels=(64,)),
            stage2=dict(num_modules=1, num_branches=2, block='BASIC', num_blocks=(4, 4), num_channels=(32, 64)),
            stage3=dict(num_modules=4, num_branches=3, block='BASIC', num_blocks=(4, 4, 4), num_channels=(32, 64, 128)),
            stage4=dict(num_modules=3, num_branches=4, block='BASIC', num_blocks=(4, 4, 4, 4), num_channels=(32, 64, 128, 256))
        ),
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://msra/hrnetv2_w32')
    ),
    decode_head=dict(
        in_channels=[32, 64, 128, 256], # Default for HRNet-W32
        num_classes=num_classes, 
        #out_channels=num_classes,
        ignore_index=255,
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            # [Background, Cracks, Alligator, Severe]
            class_weight=[0.1] + [1.0] * (num_classes - 1) + [0.1]
        ),
    )
)


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
        max_keep_ckpts=3, 
        save_best='mIoU',
        out_dir=f'{work_dir}/checkpoints',
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(
        type='SegVisualizationHook', 
        draw=True,
        interval=10
    )
)

vis_backends = [dict(type='LocalVisBackend')]

visualizer = dict(
    type='SegLocalVisualizer', 
    vis_backends=vis_backends, 
    save_dir=f'{work_dir}/results',
    name='visualizer',
    alpha=0.6
)

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
            dict(type='GaussianBlur', blur_limit=(3, 5), p=1.0),
            # Variance limit controls the severity of the noise.
            dict(type='GaussNoise', var_limit=(10.0, 50.0), p=1.0),
        ],
        p=0.5  # 50% chance to apply either Blur or Noise. 50% chance to do nothing (Identity).
    )
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False), # Crucial fix
    dict(type='RandomResize', scale=(2048, 512), ratio_range=(0.5, 2.0), keep_ratio=True),
    
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
    batch_size=batch_size, # Safety for your 4070
    num_workers=num_workers,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='splits/train.txt',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        data_prefix=dict(img_path='images', seg_map_path=label_dir),
        metainfo=metainfo,
        pipeline=train_pipeline,
        reduce_zero_label=False
    )
)

test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='splits/test.txt',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        data_prefix=dict(img_path='images', seg_map_path=label_dir),
        metainfo=metainfo,
        pipeline=test_pipeline,
        reduce_zero_label=False
    )
)

val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='splits/val.txt',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        data_prefix=dict(img_path='images', seg_map_path=label_dir),
        metainfo=metainfo,
        pipeline=test_pipeline,
        reduce_zero_label=False
    )
)

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
