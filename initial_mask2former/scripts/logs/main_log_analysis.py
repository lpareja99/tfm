import argparse
from pathlib import Path
from organize_logs import organize_and_clean
from log_extractor import parse_training_log, create_plot, parse_testing_log

def master_process(work_dir):
    root = Path(work_dir)
    
    # 1. Organize
    print(f"--- Step 1: Organizing {work_dir} ---")
    organize_and_clean(work_dir)
    
    # 2. Setup central analysis folder
    base_results_dir = root / "results" / "analysis"
    base_results_dir.mkdir(parents=True, exist_ok=True)
    
    # Process TRAINING
    train_dir = root / "logs" / "training"
    if train_dir.exists():
        runs = [f for f in train_dir.iterdir() if f.is_dir()]
        print(f"\n--- Step 2: Training Analysis (Found {len(runs)} folders) ---")
        for run_folder in runs:
            # FIX: rglob ensures we find logs inside timestamped subfolders
            log_file = next(run_folder.rglob("*.log"), None)
            if log_file:
                run_id = run_folder.name
                data = parse_training_log(str(log_file), str(base_results_dir), run_id)
                if data:
                    print(f"📈 Creating plots for {run_id}...")
                    create_plot(data, str(base_results_dir), run_id)
                else:
                    print(f"ℹ️ {run_id}: Log found, but no evaluation tables yet (Wait for first 750 iters).")
            else:
                print(f"⚠️ {run_folder.name}: No .log file found.")

    # Process TESTING
    test_dir = root / "logs" / "testing"
    if test_dir.exists():
        runs = [f for f in test_dir.iterdir() if f.is_dir()]
        print(f"\n--- Step 3: Testing Analysis (Found {len(runs)} folders) ---")
        for run_folder in runs:
            log_file = next(run_folder.rglob("*.log"), None)
            if log_file:
                run_id = run_folder.name
                print(f"📊 Analyzing Test results for {run_id}...")
                parse_testing_log(str(log_file), str(base_results_dir), run_id)
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('work_dir')
    master_process(parser.parse_args().work_dir)