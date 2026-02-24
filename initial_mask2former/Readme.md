# Table of Contents
1. [Experiment Dashboard](#Experiment-Dashboard)
2. [Classes](#Classes)
3. [Third Example](#third-example)
4. [Fourth Example](#fourth-examplehttpwwwfourthexamplecom)


## 📊 Experiment Dashboard

| ID | Model | Data | Strategy | mIoU | Status | Date|
| :---                                 | :---   | :---     | :---              | :---   | :---        | :--- |
| [**01**](./docs/experiments.md#exp1) | Swin-T | 500 Img  | Balanced 17-class | 0.XX   | Done        |  |
| [**02**](./docs/experiments.md#exp2) | Swin-T | 1500 Img | Binary Cracks     | 73.36% | Done        |  |
|[ **03**](./docs/experiments.md#exp3) | Swin-T | 1500 Img | Binary Cracks     | 73.36% | In Progress |  |

---

## Classes
| Class ID | Class Name | RGB Color Palette | Hex (Approx) |
| :--- | :--- | :--- | :--- |
| 0 | bg | (0, 0, 0) | #000000 |
| 1 | cracks | (250, 50, 83) | #FA3253 |
| 2 | cracks_alligator | (36, 179, 83) | #24B353 |
| 3 | cracks_severe | (102, 255, 102) | #66FF66 |
| 4 | edge_breaks | (255, 0, 255) | #FF00FF |
| 5 | fretting | (204, 153, 51) | #CC9933 |
| 6 | pothole | (115, 51, 128) | #733380 |
| 7 | manhole | (34, 62, 209) | #223ED1 |
| 8 | patched | (63, 63, 63) | #3F3F3F |
| 9 | bad_joint | (224, 68, 45) | #E0442D |
| 10 | joint | (255, 153, 51) | #FF9933 |
| 11 | large_repair | (255, 255, 51) | #FFFF33 |
| 12 | loose_stones | (51, 255, 255) | #33FFFF |
| 13 | pole_shadow | (172, 84, 109) | #AC546D |
| 14 | sill | (36, 223, 0) | #24DF00 |
| 15 | tyre_mark | (170, 68, 22) | #AA4416 |
| 16 | edge_grass | (213, 164, 25) | #D5A419 |

## Start Project

1. Build img `docker compose up -d --build` (~10min)

2. Start container witout building `docker compose up -d`

3. Access Container `docker exec -it road_defect_tfm bash`

4. Train data `mim train mmseg configs/local/initial_test_flowity.py`

5. Test Data 
` mim test mmseg configs/initial_test_flowity.py     --checkpoint work_dirs/initial_test_flowity/checkpoints/initial_test_flowity/best_mIoU_iter_1440.pth     --show-dir work_dirs/initial_test_flowity/results`

Option to only have to pass the work-directory
`python scripts/mim_test_executer.py work_dirs/combined_cracks_augmentation`

6. Organize and analyze training and test logs (after both training and testing have run successfully)
`python scripts/master_analysis.py work_dirs/combined_cracks_augmentation`

7. Upload the hyperparameters to tensor board
``
8. Download a mmsegmetation model 
`mim download mmsegmentation --config mask2former_swin-t_8xb2-160k_ade20k-512x512 --dest ./models/`

---

## Azure Comads

1. Build img
``` bash
export DOCKER_BUILDKIT=1
docker build --tag crflowityartifacts.azurecr.io/roadai/laura_tfm:mask2formerSwing -f ./Dockerfile .
```

2. loging to azure 
``` bash
az acr login -n crflowityartifacts
```

3. Push img: 
``` bash 
docker push crflowityartifacts.azurecr.io/roadai/laura_tfm:mask2formerSwing 
```

4. run job 
``` bash 
az ml job create --subscription 2dcd4ebb-39e0-451f-9dcb-9a3ec70e0299 --resource-group rg-flowityanalytics-testing --workspace-name ml-analytics-testing --file ./scripts/azure/train_job.yml
```

5. create and update db
``` bash
az ml data create --file scripts/azure/db_creation.yml \
  --subscription 2dcd4ebb-39e0-451f-9dcb-9a3ec70e0299 \
  --resource-group rg-flowityanalytics-testing \
  --workspace-name ml-analytics-testing
```

python scripts/tsne_analysis.py \
    work_dirs/cracks_augmentation/cracks_augmentation.py \
    work_dirs/cracks_augmentation/checkpoints/best_mIoU_iter_5000.pth \
    data/multi_crack \
    work_dirs/cracks_augmentation/results/analysis

tse analysis 

```python

python scripts/crack_tse_analysis.py \
    work_dirs/combined_cracks_augmentation/combined_cracks_augmentation.py \
    work_dirs/combined_cracks_augmentation/checkpoints/combined_cracks_augmentation/best_mIoU_iter_7000.pth \
    data/combine_crack \
    work_dirs/combined_cracks_augmentation/results/analysis

```


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