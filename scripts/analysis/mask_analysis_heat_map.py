import numpy as np
import matplotlib.pyplot as plt
import cv2

def visualize_crack_differences(y_true, y_pred, image_path=None):
    """
    y_true: Ground truth mask (2D array, 0 or 1)
    y_pred: Predicted mask (2D array, 0 or 1)
    image_path: Optional path to original image for overlay
    """
    
    # Ensure inputs are binary (0s and 1s)
    y_true = (y_true > 0).astype(np.uint8)
    y_pred = (y_pred > 0).astype(np.uint8)

    # 1. Calculate TP, FP, FN
    # True Positive (Intersection): Both refer to crack
    tp = np.logical_and(y_pred == 1, y_true == 1)
    
    # False Positive (Over-prediction): Pred is 1, Truth is 0
    fp = np.logical_and(y_pred == 1, y_true == 0)
    
    # False Negative (Missed): Pred is 0, Truth is 1
    fn = np.logical_and(y_pred == 0, y_true == 1)

    # 2. Create the RGB Error Map
    # Initialize a black image with 3 channels (R, G, B)
    h, w = y_true.shape
    error_map = np.zeros((h, w, 3), dtype=np.uint8)

    # Assign Colors
    # Green for Correct (TP)
    error_map[tp] = [0, 255, 0] 
    # Red for Noise/Over-prediction (FP)
    error_map[fp] = [255, 0, 0]
    # Blue for Missed Cracks (FN)
    error_map[fn] = [0, 0, 255]

    # 3. Visualization
    plt.figure(figsize=(15, 5))

    # Plot 1: The Error Map alone (Best for analyzing structure)
    plt.subplot(1, 2, 1)
    plt.imshow(error_map)
    plt.title("Difference Map\nGreen=Correct, Red=Extra, Blue=Missed")
    plt.axis('off')

    # Plot 2: Overlay on original image (if provided)
    if image_path:
        original_img = cv2.imread(image_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        plt.subplot(1, 2, 2)
        plt.imshow(original_img)
        # Overlay the error map with transparency (alpha=0.5)
        plt.imshow(error_map, alpha=0.5)
        plt.title("Overlay on Road Surface")
        plt.axis('off')

    plt.show()

# Example Usage:
# visualize_crack_differences(ground_truth_mask, predicted_mask, 'road_image.jpg')