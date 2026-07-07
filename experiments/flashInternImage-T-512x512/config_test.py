# ===========================================================================
# STANDALONE test/inference config for FlashInternImage (§4.2 adverse weather).
# Copy of config.py (FlashInternImage Tiny channels=64 -> matches the seed-777
# checkpoint) with train/val/test repointed to the weather dataset:
#   data_root=/app/data/final_dataset, label_dir=labels_720, splits/all_test.txt
# test_dataloader is overridden per condition (dry/wet/half) by the runner.
# ===========================================================================
_base_ = ['mmseg::mask2former/mask2former_r50_8xb2-160k_ade20k-512x512.py']

custom_imports = dict(imports=['custom_modules.backbone.flash_intern_image'], allow_failed_imports=False)

data_root = "/app/data/final_dataset"
label_dir = "labels_720"
log_level = 'INFO'
work_dir = './work_dirs/flash_weather_test'
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
batch_size = 2
num_workers = 1
val_interval = 1000
log_interval = 100

iter_per_epoch = img_num / batch_size
epochs = 80
max_iterations = int(iter_per_epoch * epochs)
crop_size = (512, 512)

# 1. Model Config
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='FlashInternImage',
        _delete_=True,
        core_op='DCNv4',
        channels=64,
        depths=[4, 3, 21, 4], # Tiny version
        groups=[4, 8, 16, 32],
        mlp_ratio=4.,
        drop_path_rate=0.2,
        norm_layer='LN',
        layer_scale=1.0,
        offset_scale=1.0,
        post_norm=False,
        with_cp=False,
        out_indices=(0, 1, 2, 3),
        init_cfg=dict(type='Pretrained', checkpoint='data/pretrained/flash_internimage_t_1k_224.pth')
    ),
    decode_head=dict(
        type='Mask2FormerHead',
        in_channels=[64, 128, 256, 512], # Default for InternImage-T
        num_classes=num_classes,
        ignore_index=255,
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[0.1] + [1.0] * (num_classes - 1) + [0.1]
        ),
    )
)


custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='mIoU',
        rule='greater',
        min_delta=0.003,
        patience=5,
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

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]

train_pipeline = test_pipeline  # not training; only so the loader can be built

train_dataloader = dict(
    batch_size=1,
    num_workers=num_workers,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='splits/all_test.txt',
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
        ann_file='splits/all_test.txt',
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
        ann_file='splits/all_test.txt',
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
    type='IterBasedTrainLoop',
    max_iters=max_iterations,
    val_interval=val_interval
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='PolyLR', begin=0, end=max_iterations, power=0.9, by_epoch=False)
]

randomness = dict(seed=None, deterministic=False)
