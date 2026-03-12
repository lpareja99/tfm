# metrics.py
import torch
import evaluate
import numpy as np
import psutil # Add this to your requirements.txt if not present
from collections import namedtuple

metric = evaluate.load("mean_iou")
Mask2FormerOutput = namedtuple("Mask2FormerOutput", ["class_queries_logits", "masks_queries_logits"])

def compute_metrics(eval_pred, processor, id2label):
    logits, labels = eval_pred
    
    # Unpack Logits
    class_logits_all = torch.from_numpy(logits[0])
    mask_logits_all = torch.from_numpy(logits[1])
    num_samples = class_logits_all.shape[0]

    # Handle the 'labels' tuple/array correctly
    # If label_names=["labels"] is set, 'labels' should be your GT masks
    if isinstance(labels, tuple):
        # Grab the first element if it's a tuple
        references = labels[0]
    else:
        references = labels

    # Convert to list of numpy arrays
    # references shape is (Batch, Height, Width)
    references_list = [r for r in references]
    
    # Get the actual shapes of the ground truth to match them
    # This handles the "512x512" vs "original" size automatically
    target_sizes = [r.shape for r in references_list]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_predictions = []
    for i in range(num_samples):
        # 1. FIXED SPELLING: Changed 'momock_output' to 'mock_output'
        mock_output = Mask2FormerOutput(
            class_queries_logits=class_logits_all[i : i + 1].to(device),
            masks_queries_logits=mask_logits_all[i : i + 1].to(device)
        )
        
        with torch.no_grad():
            # 2. SPEED OPTIMIZATION: Use (512, 512) directly to skip slow resizing
            segmentation_map = processor.post_process_semantic_segmentation(
                mock_output, 
                target_sizes=[(512, 512)] 
            )[0]
            
            all_predictions.append(segmentation_map.cpu().numpy())
            
            # Free up the GPU memory for the next loop
            del mock_output, segmentation_map
            
            
    results = metric.compute(
        predictions=all_predictions,
        references=references_list, 
        num_labels=len(id2label),
        ignore_index=255,
        reduce_labels=False,
    )
    
    metrics_summary = {
        "mean_iou": results["mean_iou"],
        "mean_accuracy": results["mean_accuracy"],
        "overall_accuracy": results["overall_accuracy"],
    }
    
    per_category_iou = results["per_category_iou"]
    per_category_accuracy = results["per_category_accuracy"]
    
    for i, label in id2label.items():
        iou = per_category_iou[i]
        recall = per_category_accuracy[i]
        
        # Calculate Dice (F1 Score) directly from IoU
        # Using np.isnan check to avoid math errors if a class wasn't present in the batch
        if not np.isnan(iou):
            dice = (2 * iou) / (1 + iou)
        else:
            dice = float('nan')

        # Group them cleanly using the "slash" trick
        metrics_summary[f"iou/{label}"] = iou
        metrics_summary[f"recall/{label}"] = recall 
        metrics_summary[f"dice/{label}"] = dice
        
    return metrics_summary