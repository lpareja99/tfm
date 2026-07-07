import os
import cv2
import numpy as np
import pandas as pd
from mmengine.config import Config
from sklearn.metrics import confusion_matrix

def evaluate_segmentation(y_true_mask, y_pred_mask, num_classes):
    """
    y_true_mask: Ground truth mask (NumPy array)
    y_pred_mask: Model prediction mask (NumPy array)
    """
    # Flatten arrays to treat every pixel as a data point
    y_true = y_true_mask.flatten()
    y_pred = y_pred_mask.flatten()

    # 1. Generate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    
    # 2. Calculate IoU per class
    # IoU = TP / (TP + FP + FN)
    intersection = np.diag(cm)
    ground_truth_set = cm.sum(axis=1)
    predicted_set = cm.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    
    # Avoid division by zero
    iou = intersection / union.astype(float)
    
    return cm, iou