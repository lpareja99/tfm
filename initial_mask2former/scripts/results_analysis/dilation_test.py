import cv2
import numpy as np
import argparse
from pathlib import Path

def calculate_iou(pd, gt):
    # Convert to binary
    intersection = np.logical_and(pd > 0, gt > 0).sum()
    union = np.logical_or(pd > 0, gt > 0).sum()
    return (intersection / union) * 100 if union > 0 else 0

def run_dilation_test(pred_dir, gt_dir, max_iters=5):
    p_path, g_path = Path(pred_dir), Path(gt_dir)
    preds = sorted(list(p_path.glob("*.png")))
    kernel = np.ones((3,3), np.uint8)

    if not preds:
        return print(f"❌ No masks found in {pred_dir}")

    print(f"🧪 Comparing {len(preds)} predicted masks against dilated ground truth...")
    
    # results[iteration] = [score1, score2, ...]
    results = {i: [] for i in range(max_iters + 1)}

    for p_file in preds:
        g_file = g_path / p_file.name
        if not g_file.exists(): continue
        
        p = cv2.imread(str(p_file), 0)
        g = cv2.imread(str(g_file), 0)
        
        # Iteration 0 is the standard IoU
        results[0].append(calculate_iou(p, g))
        
        # Gradually make the Ground Truth fatter
        temp_g = g.copy()
        for i in range(1, max_iters + 1):
            temp_g = cv2.dilate(temp_g, kernel, iterations=1)
            results[i].append(calculate_iou(p, temp_g))

    print("\n--- Dilation Test Results ---")
    print(f"Standard mIoU (0px fat):  {np.mean(results[0]):.2f}%")
    for i in range(1, max_iters + 1):
        avg = np.mean(results[i])
        diff = avg - np.mean(results[0])
        print(f"Dilated +{i}px mIoU:       {avg:.2f}% ({'+' if diff >= 0 else ''}{diff:.2f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('pred_dir', help="Path to your predicted masks/")
    parser.add_argument('gt_dir', help="Path to your ground truth labels/")
    args = parser.parse_args()
    run_dilation_test(args.pred_dir, args.gt_dir)