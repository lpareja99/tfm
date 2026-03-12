import subprocess, argparse, os
from pathlib import Path

def run_azure_test(config, checkpoint, data_root, out_dir):
    # Ensure the output directory exists
    os.makedirs(out_dir, exist_ok=True)
    
    # We use the standard 'mim test' command
    # but we inject the dynamic Azure paths for the data and checkpoint
    cmd = [
        "mim", "test", "mmseg", config,
        "--checkpoint", checkpoint,
        "--show-dir", out_dir,
        "--cfg-options", f"data_root={data_root}", # Force the test to use Azure's data path
        "--out", os.path.join(out_dir, "masks")
    ]

    print(f"Testing config: {config} | Using Checkpoint: {checkpoint}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--out_dir', required=True)
    args = parser.parse_args()
    
    run_azure_test(args.config, args.checkpoint, args.data_root, args.out_dir)