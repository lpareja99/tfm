<a name="exp3"></a>
## Exp 1: Intern Image Backbone Basic

**Date:** 2026-03-11

#### Instructions

* **Download InternImage backbone:**
    ```bash
    wget [https://raw.githubusercontent.com/OpenGVLab/InternImage/master/segmentation/mmseg_custom/models/backbones/intern_image.py](https://raw.githubusercontent.com/OpenGVLab/InternImage/master/segmentation/mmseg_custom/models/backbones/intern_image.py) -O /app/custom_modules/intern_image.py
    ```

* **Download and extract DCNv3:**
    ```bash
    wget [https://github.com/OpenGVLab/InternImage/archive/refs/heads/master.zip](https://github.com/OpenGVLab/InternImage/archive/refs/heads/master.zip) -O master.zip
    unzip master.zip "InternImage-master/segmentation/ops_dcnv3/*" -d /app/custom_modules/
    mv /app/custom_modules/InternImage-master/segmentation/ops_dcnv3 /app/custom_modules/ops_dcnv3
    rm -rf /app/custom_modules/InternImage-master master.zip
    ```

* **Install build dependencies:**
    ```bash
    pip install ninja
    pip install timm
    ```

* **Configure CUDA environment:**
    ```bash
    export CUDA_HOME=/usr/local/cuda
    export CPATH=$CUDA_HOME/include:$CPATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    ```

* **Build CUDA operators:**
    ```bash
    sh make.sh
    ```

* **Adjust Code:** Update codebase of InternImage backbone for compatibility with the new version of `mmsegmentation`.

* Train needs to be pass with `PYTHONPATH="." mim train mmseg configs/local/InternImage.py`

**Config:** `configs/azure/cracks_augmentation.py`  
**Work Dir:** `work_dirs/azure/cracks_augmentation_24_02`


#### 1. Hypothesis / Goal

* **Goal:**
* **Hypothesis:**


#### 2. Key Hyperparameters

* **Model:** Maks2Former with Swing Tiny, (512,512)
* **Iterations:** 10500
* **Batch Size:**
* **Classes:** 4 
* **Data Selection:** 



#### 3. Results (mIoU / Metrics)

* **Best mIoU:**
* **Class Performance:**

    * **Failed:**

    * **Success:**



#### 4. Observations & Next Steps

* **Observation:**

* Speed Training(Local): ~0.44s

---


<a name="exp3"></a>
## Exp : Flash Intern Image Backbone Basic

**Date:** 2026-03-11

#### Instructions

* **Download and extract DCNv4:**
    ```bash
    wget https://github.com/OpenGVLab/DCNv4/archive/refs/heads/main.zip -O main.zip
    unzip main.zip "DCNv4-main/DCNv4_op/*" -d /app/custom_modules/
    mv /app/custom_modules/DCNv4-main/DCNv4_op /app/custom_modules/ops_dcnv4
    rm -rf /app/custom_modules/DCNv4-main main.zip
    ```

* **Install build dependencies:**
    ```bash
    pip install ninja
    pip install timm
    ```

* **Configure CUDA environment:**
    ```bash
    export CUDA_HOME=/usr/local/cuda
    export CPATH=$CUDA_HOME/include:$CPATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    ```

* **Build CUDA operators:**
    ```bash
    sh make.sh
    ```

* **Adjust Code:** Update InterImage backbone to use DCNv4.

* Train needs to be pass with `PYTHONPATH="." mim train mmseg configs/local/InternImage.py`

**Config:** `configs/azure/cracks_augmentation.py`  
**Work Dir:** `work_dirs/azure/cracks_augmentation_24_02`


#### 1. Hypothesis / Goal

* **Goal:**
* **Hypothesis:**


#### 2. Key Hyperparameters

* **Model:** Maks2Former with Swing Tiny, (512,512)
* **Iterations:** 10500
* **Batch Size:**
* **Classes:** 4 
* **Data Selection:** 



#### 3. Results (mIoU / Metrics)

* **Best mIoU:**
* **Class Performance:**

    * **Failed:**

    * **Success:**



#### 4. Observations & Next Steps

* **Observation:**

* Speed Trainig (Local): ~0.36s (15% faster tha InternImage)

---


