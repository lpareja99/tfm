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

