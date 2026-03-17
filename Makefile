EXP ?= 1a

var_1a = 1a_swin_t_cracks.vars
var_1b = 1b_swin_t_all_defects.vars
var_1c = 1c_swin_t_all_defects_relabel.vars

var_2a = 2a_hrnet_t_all_defects.vars

include runs/$(var_$(EXP))

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


# ==========================================
# test
# ==========================================
test:
	python initial_mask2former/scripts/mim_test_executer.py \
		--work-dir $(WORK_DIR) \
		--data-root $(LOCAL_DATA) \
		--label-dir $(LABEL)


