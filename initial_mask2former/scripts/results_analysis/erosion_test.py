import cv2
import numpy as np
import argparse
from pathlib import Path

def calculate_iou(pd, gt):
    intersection = np.logical_and(pd > 0, gt > 0).sum()
    union = np.logical_or(pd > 0, gt > 0).sum()
    return (intersection / union) * 100 if union > 0 else 0

def run_erosion_test(pred_dir, gt_dir, max_iters=5):
    p_path, g_path = Path(pred_dir), Path(gt_dir)
    preds = sorted(list(p_path.glob("*.png")))
    kernel = np.ones((3,3), np.uint8)

    print(f"🧪 Comparing {len(preds)} masks against ground truth...")
    
    # Store results for different thinning levels
    results = {i: [] for i in range(max_iters + 1)}

    for p_file in preds:
        g_file = g_path / p_file.name
        if not g_file.exists(): continue
        
        p = cv2.imread(str(p_file), 0)
        g = cv2.imread(str(g_file), 0)
        
        # Calculate IoU for original and thinned versions
        results[0].append(calculate_iou(p, g))
        for i in range(1, max_iters + 1):
            g = cv2.erode(g, kernel, iterations=1)
            results[i].append(calculate_iou(p, g))

    print("\n--- Final Erosion Results ---")
    print(f"Standard mIoU (0px thin): {np.mean(results[0]):.2f}%")
    for i in range(1, max_iters + 1):
        avg = np.mean(results[i])
        diff = avg - np.mean(results[0])
        print(f"Thinned {i}px mIoU:       {avg:.2f}% (+{diff:.2f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('pred_dir', help="Path to your masks/ folder")
    parser.add_argument('gt_dir', help="Path to your data labels/ folder")
    args = parser.parse_args()
    run_erosion_test(args.pred_dir, args.gt_dir)