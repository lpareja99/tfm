import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str)
parser.add_argument('--data_root', type=str)
parser.add_argument('--labels_folder', type=str)
parser.add_argument('--weights', type=str)
parser.add_argument('--work_dir', type=str)
args = parser.parse_args()

cmd_parts = [
    "mim train mmseg",
    args.config,
    f"--work-dir={args.work_dir}",
    f"--cfg-options",
    f"data_root={args.data_root}",
    f"train_dataloader.dataset.data_root={args.data_root}",
    f"val_dataloader.dataset.data_root={args.data_root}",
    f"test_dataloader.dataset.data_root={args.data_root}",
    f"train_dataloader.dataset.data_prefix.seg_map_path={args.labels_folder}",
    f"val_dataloader.dataset.data_prefix.seg_map_path={args.labels_folder}",
    f"test_dataloader.dataset.data_prefix.seg_map_path={args.labels_folder}",
    f"default_hooks.checkpoint.out_dir={args.work_dir}/checkpoints",
    f"val_evaluator.output_dir='{args.work_dir}/eval_results'",
    f"test_evaluator.output_dir='{args.work_dir}/eval_results'",
    f"load_from={args.weights}",
]

train_cmd = " ".join(cmd_parts)
print(f"Executing: {train_cmd}")
os.system(train_cmd)