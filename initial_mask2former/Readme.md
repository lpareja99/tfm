1. Build img `docker compose up -d --build` (~10min)

2. Start container witout building `docker compose up -d`

3. Access Container `docker exec -it road_defect_tfm bash`

4. Train data `mim train mmseg configs/initial_test_flowity.py`

5. Test Data 
` mim test mmseg configs/initial_test_flowity.py     --checkpoint work_dirs/initial_test_flowity/checkpoints/initial_test_flowity/best_mIoU_iter_1440.pth     --show-dir work_dirs/initial_test_flowity/results`



6. Loss Curve
`mim run mmseg analyze_logs plot_curve \
work_dirs/initial_test_flowity/20260213_112247/20260213_112247.json \
--keys loss --legend loss --out loss_curve.png`

7. mIoU Curve


// test_visual -- following mmsegmentation docuemnantation: [text](https://mmsegmentation.readthedocs.io/en/latest/user_guides/visualization.html)

pip install tensorboardX
pip install future tensorboard

vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

mim train mmseg configs/initial_test_flowity.py --work-dir work_dirs/test_visual

tensorboard --logdir work_dirs/test_visual/vis_data/20260213_130959

tensorboard --logdir work_dirs --port 6006 --bind_all


## Experiments

<details>
<summary><h3>Exp 1: Swin-T Mask2Former - Initial 500-Image Balanced Subset</h3></summary>

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
</details>

---

<details>
<summary><h3>Exp 2: Swin-T Mask2Former - 1500 Img Subset with only cracks</h3></summary>

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

</details>

---

<summary><h3>Exp 2: Swin-T Mask2Former - 1500 Img Subset with only cracks with augmentation techniques</h3></summary>

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

</details>


