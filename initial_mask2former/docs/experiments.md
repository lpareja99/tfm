# Experiments

<a name="exp1"></a>
## Exp 1: Swin-T Mask2Former - Initial 500-Image Balanced Subset

**Date:** 2026-02-17  
**Config:** `configs/subset_500_flowity_bg.py`  
**Work Dir:** `work_dirs/subset_500_flowity_bg/20260217_092814`

#### 1. Hypothesis / Goal
* **Goal:** Test if a small, balanced subset of 500 images using a greedy selection approach can provide a baseline for 17 road defect classes to reduce bias.
* **Hypothesis:** By ensuring at least 40-67 instances per class, the model should begin to distinguish between primary defects like cracks and potholes.

#### 2. Key Hyperparameters
* **Model:** Swin-T Mask2Former (512x512).
* **Iterations:** 15,000.
* **Batch Size:** 2.
* **Classes:** 17 (including background).
* **Data Selection:** Greedy approach targeting uniform distribution; excluded IDs 14, 9, 16 (<20 instances).

#### 3. Results (mIoU / Metrics)
* **Best mIoU:** 0.XX (at iter 13500).
* **Class Performance:** * **Failed:** Class 1 ("cracks") failed to learn entirely despite having 67 instances.
    * **Success:** Better performance on distinct geometric shapes like "pothole".

#### 4. Observations & Next Steps
* **Observation:** The "crack" class is likely struggling due to its thin, linear nature and high intra-class variance, which 500 images cannot represent.

---

<a name="exp2"></a>
## Exp 2: Swin-T Mask2Former - 1500 Img Subset with only cracks

**Date:** 2026-02-17  
**Config:** `combined_cracks_bg.py`  
**Work Dir:** `./work_dirs/combined_cracks_bg`

#### 1. Hypothesis / Goal
* **Goal:** Establish a robust baseline for crack segmentation using a Transformer-based architecture (Mask2Former) on only cracks merging their classes.

* **Hypothesis:** If I reduce the numebr of classes and merge them into one the model would not struggle so much to learn the pattern.

#### 2. Key Hyperparameters
* **Model:** Mask2Former with Swin-Tiny backbone
* **Iterations:** 5 Epochs
* **Batch Size:** 2
* **Classes:** 2 (gb, cracks)
* **Data Selection:** 1500 img subset

#### 3. Results (mIoU / Metrics)
* **Best mIoU:** 73.36% (Final Test Result)
* **Class Performance:** 
    * **Failed:** Cracks achieved only 48.98% IoU
    * **Success:** 71.31% Recall

#### 4. Observations & Next Steps
* **Observation:** model effectively identifies the presence of cracks (High Recall) but struggles with the precision of the crack boundaries, resulting in the "dotted line" effect seen in your test images.


---

<a name="exp3"></a>
## Exp 3: Swin-T Mask2Former - 1500 Img Subset with only cracks with augmentation techniques

**Date:** 2026-02-19

**Config:** ``  

**Work Dir:** ``



#### 1. Hypothesis / Goal

* **Goal:**

* **Hypothesis:**



#### 2. Key Hyperparameters

* **Model:**

* **Iterations:**

* **Batch Size:**

* **Classes:**

* **Data Selection:**



#### 3. Results (mIoU / Metrics)

* **Best mIoU:**

* **Class Performance:**

    * **Failed:**

    * **Success:**



#### 4. Observations & Next Steps

* **Observation:**

---