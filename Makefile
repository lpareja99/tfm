EXP ?= 1a

var_1a = 1a_swin_t_cracks.vars
var_1b = 1b_swin_t_all_defects.vars
var_1c = 1c_swin_t_all_defects_relabel.vars

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
		--file mask2former/swin-T-512x512/azure/train_job.yml \
		--subscription 2dcd4ebb-39e0-451f-9dcb-9a3ec70e0299 \
		--resource-group rg-flowityanalytics-testing \
		--workspace-name ml-analytics-testing


# ==========================================
# test
# ==========================================
test:
	python initial_mask2former/scripts/mim_test_executer.py \
		--work-dir $(WORK_DIR_1A) \
		--data-root $(LOCAL_DATA_1A) \
		--label-dir $(LABEL_1A)


# ==========================================================
# EXPERIMENT 1B: Swin-T-512x512 - All Defects No Relabeling
# ==========================================================
LOCAL_DATA_2A = /app/data/2026-01-19-defect_dataset
LABEL_2A = label_basic_defects
WORK_DIR_2A = mask2former/swin-T-512x512/output/local/test_run
CLASSES_2A = ("bg","cracks","cracks_alligator","cracks_severe","edge_breaks","fretting","pothole","manhole","patched","bad_joint","joint","large_repair","loose_stones","pole_shadow","sill","tyre_mark","edge_grass")
COLORS_2A = [(0,0,0), (250,50,83), (36,179,83), (102,204,255)]
WEIGHTS_2A = [0.1,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.1]
NUM_CLASSES_2A = 9

train-local-cracks2:
	mim train mmseg mask2former/swin-T-512x512/cracks_augmentation.py \
		--work-dir $(WORK_DIR_2A) \
		--cfg-options \
		data_root=$(LOCAL_DATA_2A) \
		train_dataloader.dataset.data_root=$(LOCAL_DATA_2A) \
		val_dataloader.dataset.data_root=$(LOCAL_DATA_2A) \
		test_dataloader.dataset.data_root=$(LOCAL_DATA_2A) \
		train_dataloader.dataset.data_prefix.seg_map_path=$(LABEL_2A) \
		val_dataloader.dataset.data_prefix.seg_map_path=$(LABEL_2A) \
		test_dataloader.dataset.data_prefix.seg_map_path=$(LABEL_2A) \
		train_dataloader.batch_size=2 \
		train_dataloader.num_workers=4 \
		train_cfg.max_iters=500 \
		train_cfg.val_interval=100 \
		custom_hooks.0.min_delta=0.05 \
		custom_hooks.0.patience=1 \
		visualizer.save_dir=$(WORK_DIR_2A)/results \
		default_hooks.checkpoint.out_dir=$(WORK_DIR_2A)/checkpoints \
		model.decode_head.num_classes=$(NUM_CLASSES_2A) \
		model.decode_head.out_channels=$(NUM_CLASSES_2A) \
		model.decode_head.loss_cls.class_weight=$(WEIGHTS_2A) \
		train_dataloader.dataset.metainfo.classes="$(CLASSES_2A)" \
        val_dataloader.dataset.metainfo.classes="$(CLASSES_2A)" \
		train_dataloader.dataset.metainfo.palette="$(COLORS_2A)" \
		val_dataloader.dataset.metainfo.palette="$(COLORS_2A)" \
		metainfo.classes="$(CLASSES_2A)" \
		metainfo.palette="$(COLORS_2A)"



test-cracks2:
	python initial_mask2former/scripts/mim_test_executer.py \
		--work-dir $(WORK_DIR_2A) \
		--data-root $(LOCAL_DATA_2A) \
		--label-dir $(LABEL_2A)


# ==========================================================
# EXPERIMENT 1C: Swin-T-512x512 - All Defects Relabeling
# ==========================================================
