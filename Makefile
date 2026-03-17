# ==========================================
# EXPERIMENT 1A: Swin-T-512x512 - Crack Only
# ==========================================
LOCAL_DATA_1A = /app/data/2026-01-19-defect_dataset
LABEL_1A = labels_cracks
WORK_DIR_1A = mask2former/swin-T-512x512/output/local/only_cracks_2
CLASSES_1A = ('bg','cracks','cracks_alligator','cracks_severe')
COLORS_1A = [(0,0,0), (250,50,83), (36,179,83), (102,204,255)]
WEIGHTS_1A = [0.1,1.0,1.0,1.0,0.1]
NUM_CLASSES_1A = 4

train-local-cracks:
	mim train mmseg mask2former/swin-T-512x512/cracks_augmentation.py \
		--work-dir $(WORK_DIR_1A) \
		--cfg-options \
		data_root=$(LOCAL_DATA_1A) \
		train_dataloader.dataset.data_root=$(LOCAL_DATA_1A) \
		val_dataloader.dataset.data_root=$(LOCAL_DATA_1A) \
		test_dataloader.dataset.data_root=$(LOCAL_DATA_1A) \
		train_dataloader.dataset.data_prefix.seg_map_path=$(LABEL_1A) \
		val_dataloader.dataset.data_prefix.seg_map_path=$(LABEL_1A) \
		test_dataloader.dataset.data_prefix.seg_map_path=$(LABEL_1A) \
		train_dataloader.batch_size=2 \
		train_dataloader.num_workers=4 \
		train_cfg.max_iters=500 \
		train_cfg.val_interval=100 \
		custom_hooks.0.min_delta=0.05 \
		custom_hooks.0.patience=1 \
		visualizer.save_dir=$(WORK_DIR_1A)/results \
		default_hooks.checkpoint.out_dir=$(WORK_DIR_1A)/checkpoints \
		model.decode_head.num_classes=$(NUM_CLASSES_1A) \
		model.decode_head.out_channels=$(NUM_CLASSES_1A) \
		model.decode_head.loss_cls.class_weight=$(WEIGHTS_1A) \
		train_dataloader.dataset.metainfo.classes="$(CLASSES_1A)" \
        val_dataloader.dataset.metainfo.classes="$(CLASSES_1A)" \
		train_dataloader.dataset.metainfo.palette="$(COLORS_1A)" \
		val_dataloader.dataset.metainfo.palette="$(COLORS_1A)" \
		metainfo.classes="$(CLASSES_1A)" \
		metainfo.palette="$(COLORS_1A)"

train-azure-cracks:
	az ml job create \
		--file mask2former/swin-T-512x512/azure/train_job.yml \
		--subscription 2dcd4ebb-39e0-451f-9dcb-9a3ec70e0299 \
		--resource-group rg-flowityanalytics-testing \
		--workspace-name ml-analytics-testing

test-cracks:
	python initial_mask2former/scripts/mim_test_executer.py \
		--work-dir $(WORK_DIR_1A) \
		--data-root $(LOCAL_DATA_1A) \
		--label-dir $(LABEL_1A)

# ==========================================================
# EXPERIMENT 1B: Swin-T-512x512 - All Defects No Relabeling
# ==========================================================
LOCAL_DATA_2A = /app/data/2026-01-19-defect_dataset
LABEL_2A = "label_cracks"


# ==========================================================
# EXPERIMENT 1C: Swin-T-512x512 - All Defects Relabeling
# ==========================================================
