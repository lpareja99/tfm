import subprocess, argparse
from pathlib import Path

def run_auto_test(work_dir, data_root, label_dir):
    root = Path(work_dir)
    
    config = next(root.glob("*.py"), None)
    ckpts = sorted(list(root.rglob("best_mIoU_iter_*.pth")))
    
    if not config or not ckpts:
        return print(f"Missing files in {work_dir}. Check for .py and best_mIoU .pth")

    mask_dir = root / "results" / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "mim", "test", "mmseg", str(config),
        "--checkpoint", str(ckpts[-1]),
        "--show-dir", str(root / "results"),
        "--out", str(mask_dir),
        "--work-dir", str(root),
        "--cfg-options",
        f"data_root={data_root}",
        f"test_dataloader.dataset.data_root={data_root}",
        f"test_dataloader.dataset.data_prefix.seg_map_path={label_dir}",
        f"train_dataloader.dataset.data_root={data_root}",
        f"val_dataloader.dataset.data_root={data_root}"
    ]

    print(f"Best: {ckpts[-1].name} | Executing test...")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-dir', required=True, help='Path to workspace')
    parser.add_argument('--data-root', required=True, help='Path to defect dataset')
    parser.add_argument('--label-dir', required=True, help='Specific label folder')
    
    args = parser.parse_args()
    run_auto_test(args.work_dir, args.data_root, args.label_dir)