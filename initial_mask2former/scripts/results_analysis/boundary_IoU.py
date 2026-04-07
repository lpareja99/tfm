import cv2
import numpy as np
import argparse
from pathlib import Path
from scipy.spatial.distance import directed_hausdorff

def get_boundary(mask):
    # Extract just the edge of the crack
    mask_8u = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundary_pts = np.vstack([c.reshape(-1, 2) for c in contours]) if contours else np.array([])
    return boundary_pts

def run_boundary_test(pred_dir, gt_dir):
    p_path, g_path = Path(pred_dir), Path(gt_dir)
    preds = sorted(list(p_path.glob("*.png")))
    
    distances = []
    print(f"📏 Calculating Hausdorff Distance for {len(preds)} images...")

    for p_file in preds:
        g_file = g_path / p_file.name
        if not g_file.exists(): continue
        
        p_mask = cv2.imread(str(p_file), 0)
        g_mask = cv2.imread(str(g_file), 0)
        
        p_pts = get_boundary(p_mask)
        g_pts = get_boundary(g_mask)
        
        if p_pts.size > 0 and g_pts.size > 0:
            # Hausdorff is the max distance between a point in one set and the nearest point in the other
            d1 = directed_hausdorff(p_pts, g_pts)[0]
            d2 = directed_hausdorff(g_pts, p_pts)[0]
            distances.append(max(d1, d2))
        elif p_pts.size == 0 and g_pts.size == 0:
            distances.append(0) # Perfect match on background
        else:
            distances.append(100) # High penalty for missing a crack entirely or hallucinating one

    avg_dist = np.mean(distances)
    print(f"\n--- Boundary Analysis Results ---")
    print(f"Average Hausdorff Distance: {avg_dist:.2f} pixels")
    
    if avg_dist < 10:
        print("✅ Result: Low Distance. Your model is in the right spot; focus on refining width (Dice Loss).")
    else:
        print("⚠️ Result: High Distance. Your model is hallucinating or missing crack branches (OHEM/Subgroups).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('pred_dir')
    parser.add_argument('gt_dir')
    args = parser.parse_args()
    run_boundary_test(args.pred_dir, args.gt_dir)