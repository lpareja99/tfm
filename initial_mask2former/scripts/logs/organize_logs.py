import shutil
import argparse
from pathlib import Path

def get_run_status(log_path):
    content = log_path.read_text(errors='ignore')
    # Check for table headers or signs of progress
    if "+--------+-------+-------+" in content or "mIoU" in content:
        if "Iter(train)" in content or "Saving checkpoint" in content:
            return "training"
        if "Iter(test)" in content:
            return "testing"
    return "incomplete" # Instead of None, label it as incomplete

def organize_and_clean(root_dir):
    root = Path(root_dir)
    # Protected folders
    protected = {'logs', 'results', 'checkpoints', 'analysis'}
    counts = {"training": 0, "testing": 0, "incomplete": 0}

    # Only iterate through work directories, not the protected ones
    for folder in [f for f in root.iterdir() if f.is_dir() and f.name not in protected]:
        log_file = next(folder.glob("*.log"), None)
        
        # Determine status
        if log_file:
            status = get_run_status(log_file)
        else:
            status = "incomplete" # Keep even if no log file exists just in case

        # MOVE everything, never DELETE
        dest = root / "logs" / status
        dest.mkdir(parents=True, exist_ok=True)
        
        # Check if destination already exists to avoid errors
        if not (dest / folder.name).exists():
            shutil.move(str(folder), str(dest / folder.name))
            print(f"MOVED to logs/{status}: {folder.name}")
            counts[status] += 1
        else:
            print(f"SKIPPED (Already exists): {folder.name}")

    print(f"\n--- Final State: {counts['training']} Train, {counts['testing']} Test, {counts['incomplete']} Incomplete (Safe) ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir')
    organize_and_clean(parser.parse_args().root_dir)