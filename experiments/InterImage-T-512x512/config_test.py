# ===========================================================================
# Test/inference config (§4.2 adverse weather) for InterImage-T.
# Inherits the EXACT backbone/decode head from config.py (so it matches the
# trained checkpoint; the model is not re-declared, avoiding any mismatch) and
# only repoints the data to the weather dataset: data_root=final_dataset,
# seg_map=labels_720, split=all_test. run_weather.sh overrides test_dataloader
# per condition. InterImage uses DCNv3 (CUDA op) -> needs GPU + road_defect_intern.
# ===========================================================================
_base_ = ['./config.py']

data_root = "/app/data/final_dataset"
label_dir = "labels_720"

metainfo = dict(
    classes=("bg", "cracks", "cracks_alligator", "cracks_severe", "edge_cracks", "fretting", "pothole", "manhole", "pole_shadow"),
    palette=[[0, 0, 0], [250, 50, 83], [36, 179, 83], [102, 204, 255], [255, 165, 0], [128, 128, 128], [255, 255, 0], [0, 255, 255], [255, 0, 255]],
)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs'),
]
train_pipeline = test_pipeline  # not training; the train loader is only built because of the EarlyStoppingHook

test_dataloader = dict(
    batch_size=1, num_workers=1,
    dataset=dict(
        data_root=data_root, ann_file='splits/all_test.txt',
        data_prefix=dict(img_path='images', seg_map_path=label_dir),
        metainfo=metainfo, pipeline=test_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=1, num_workers=1,
    dataset=dict(
        data_root=data_root, ann_file='splits/all_test.txt',
        data_prefix=dict(img_path='images', seg_map_path=label_dir),
        metainfo=metainfo, pipeline=test_pipeline,
    ),
)
train_dataloader = dict(
    batch_size=1, num_workers=1,
    dataset=dict(
        data_root=data_root, ann_file='splits/all_test.txt',
        data_prefix=dict(img_path='images', seg_map_path=label_dir),
        metainfo=metainfo, pipeline=train_pipeline,
    ),
)
