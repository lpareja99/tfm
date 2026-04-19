EXP ?= 1a

var_1a = 1a_swin_t_cracks.env
var_1b = 1b_swin_t_all_defects.env
var_1c = 1c_swin_t_all_defects_relabel.env

var_2a = 2a_hrnet_t_all_defects.env
var_3a = 3a_intern_image.env
var_4a = 4a_flash_intern_image.env
var_5a = 5a_beit2_base.env

include runs/$(var_$(EXP))

# =============
# general
# =============
jobs = ["purple_wing_zf84jm3h37", "patient_bulb_h67htrw5yd"]
jobs2 = zen_rice_94w4vl9xbk
output_dir ?= "./data/laura_tfm_sun_22_dry_flowity_pipeline/"
input_dir ?= "./data/laura_tfm_dry_annotated/test_annotation_output"
tfm_data_output ?= "./data/final_test_dataset/"

download-job-flowity-pipeline:
	python3 initial_mask2former/scripts/azure/download_job.py --job-id $(jobs) --output-dir $(output_dir)

cvat_input_annotation_cration:
	python3 initial_mask2former/scripts/mask_to_cvat.py --input-dir $(jobs2)

cvat_output_organization_mapping:
	python3 initial_mask2former/scripts/save_cvat_output.py --cvat-dir $(input_dir) --work-dir $(output_dir)

build_test_dataset:
	python3 initial_mask2former/scripts/build_test_dataset_cvat.py --work-dir $(output_dir) --output-dir $(tfm_data_output)

# ==========================================
# train-local
# ==========================================
train-local:
	mim train mmseg $(CONFIG_FILE) \
		--work-dir $(WORK_DIR) \
		--cfg-options \
		data_root=$(LOCAL_DATA) \
		train_dataloader.dataset.data_root=$(LOCAL_DATA) \
		val_dataloader.dataset.data_root=$(LOCAL_DATA) \
		test_dataloader.dataset.data_root=$(LOCAL_DATA) \
		train_dataloader.dataset.data_prefix.seg_map_path=$(LABEL) \
		val_dataloader.dataset.data_prefix.seg_map_path=$(LABEL) \
		test_dataloader.dataset.data_prefix.seg_map_path=$(LABEL) \
		train_dataloader.batch_size=$(BATCH_SIZE) \
		train_dataloader.num_workers=$(NUM_WORKERS) \
		train_cfg.max_iters=$(MAX_ITERS) \
		train_cfg.val_interval=$(VAL_INTERVAL) \
		custom_hooks.0.min_delta=$(DELTA_INTERVAL) \
		custom_hooks.0.patience=$(PATIENCE) \
		visualizer.save_dir=$(WORK_DIR)/results \
		default_hooks.checkpoint.out_dir=$(WORK_DIR)/checkpoints \
		model.decode_head.num_classes=$(NUM_CLASSES) \
		model.decode_head.out_channels=$(NUM_CLASSES) \
		model.decode_head.loss_cls.class_weight=$(WEIGHTS) \
		train_dataloader.dataset.metainfo.classes="$(CLASSES)" \
        val_dataloader.dataset.metainfo.classes="$(CLASSES)" \
		train_dataloader.dataset.metainfo.palette="$(COLORS)" \
		val_dataloader.dataset.metainfo.palette="$(COLORS)" \
		metainfo.classes="$(CLASSES)" \
		metainfo.palette="$(COLORS)"


# ==========================================
# train-azure
# ==========================================
train-azure:
	az ml job create \
		--file $(AZURE_TRAIN) \
		--subscription 2dcd4ebb-39e0-451f-9dcb-9a3ec70e0299 \
		--resource-group rg-flowityanalytics-testing \
		--workspace-name ml-analytics-testing


test-local:
	mim test mmseg $(CONFIG_FILE) \
		--checkpoint $(CHECKPOINT_PATH) \
		--work-dir $(WORK_DIR) \
		--cfg-options \
		data_root=$(LOCAL_DATA) \
		test_dataloader.dataset.data_root=$(LOCAL_DATA) \
		test_dataloader.dataset.data_prefix.seg_map_path=$(LABEL) \
		test_dataloader.dataset.metainfo.classes="$(CLASSES)" \
		test_dataloader.dataset.metainfo.palette="$(COLORS)" \
		test_evaluator.output_dir=$(WORK_DIR)/test_results \
		visualizer.save_dir=$(WORK_DIR)/test_visuals

# ==========================================
# test
# ==========================================
test:
	python initial_mask2former/scripts/mim_test_executer.py \
		--work-dir $(WORK_DIR) \
		--data-root $(LOCAL_DATA) \
		--label-dir $(LABEL)


mim test mmseg config_tiny.py \
    --checkpoint output/best_mIoU_iter_28000_flash.pth \
    --show-dir output/test_visuals \
    --cfg-options \
    test_dataloader.dataset.data_prefix.seg_map_path=labels_basic_defects_relabel

