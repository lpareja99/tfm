import subprocess, argparse
from pathlib import Path

from sympy import root

def run_auto_test(work_dir):
    root = Path(work_dir)
    
    # Discovery: find the config and the highest iteration best checkpoint
    config = next(root.glob("*.py"), None)
    # Recursively find pth files, filter for 'best', and take the last one (latest iteration)
    ckpts = sorted(list(root.rglob("best_mIoU_iter_*.pth")))
    
    if not config or not ckpts:
        return print(f"Missing files in {work_dir}. Check for .py and best_mIoU .pth")

    mask_dir = root / "results" / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "mim", "test", "mmseg", str(config),
        "--checkpoint", str(ckpts[-1]),
        "--show-dir", str(root / "results"),
        "--out", str(mask_dir)
    ]

    print(f"Best: {ckpts[-1].name} | Executing test...")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('work_dir')
    run_auto_test(parser.parse_args().work_dir)