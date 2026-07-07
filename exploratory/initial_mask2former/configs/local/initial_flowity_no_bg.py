# initial_flowity_no_bg.py
_base_ = ['./initial_test_flowity.py'] 

work_dir = './work_dirs/flowity_no_background'

# 1. Update the Hook WITHOUT deleting the others (Logger, Timer, etc.)
# We do this by modifying only the checkpoint sub-key
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', 
        out_dir=f'{work_dir}/checkpoints', 
        save_best='mIoU'
    )
)

# 2. Update the Visualizer while explicitly keeping the parent's backends
# In MMEngine, dictionaries are replaced unless you merge them carefully
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')],
    save_dir=f'{work_dir}/results'
)

# 3. Update the Evaluator 
val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU', 'mDice', 'mFscore'],
    ignore_index=0 # Background exclusion
)
test_evaluator = val_evaluator