# ============================================================================
# Road-defect semantic segmentation (Mask2Former, 5 swappable backbones).
# Task runner — `make help` lists every target.
#
# Local training/testing use a per-experiment preset in runs/*.env, selected
# with EXP (default 1a):   make train-local EXP=2a
# Azure/ACR coordinates come from a gitignored .env (see .env.example).
# ============================================================================

# ---- experiment preset selection (train-local / test-local) ----
EXP ?= 1a
var_1a = 1a_swin_t_cracks.env
var_1b = 1b_swin_t_all_defects.env
var_1c = 1c_swin_t_all_defects_relabel.env
var_2a = 2a_hrnet_t_all_defects.env
var_3a = 3a_intern_image.env
var_4a = 4a_flash_intern_image.env
var_5a = 5a_beit2_base.env
-include runs/$(var_$(EXP))

# Azure/ACR coordinates (gitignored; see .env.example). Soft include.
-include .env

.DEFAULT_GOAL := help

# ============================================================================
# help
# ============================================================================
.PHONY: help
help:  ## Show this help
	@echo "Road-defect segmentation — make targets:"
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) \
		| sort | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Backbone presets (EXP=): 1a/1b/1c swin · 2a hrnet · 3a interimage · 4a flash · 5a beit"

# ============================================================================
# Docker images
# ============================================================================
.PHONY: build-base build-flash build-intern
build-base:  ## Build road_defect_base (swin/hrnet/beit + parent of the CUDA images)
	docker build -t road_defect_base:latest .

build-flash:  ## Build road_defect_flash (FlashInternImage — DCNv4 compiled for sm_50)
	docker build -t road_defect_flash:latest \
		-f experiments/flashInternImage-T-512x512/Dockerfile.local \
		experiments/flashInternImage-T-512x512/custom_modules/ops_dcnv4

build-intern:  ## Build road_defect_intern (InterImage — DCNv3 compiled for sm_50)
	docker build -t road_defect_intern:latest \
		-f experiments/InterImage-T-512x512/Dockerfile.local .

# ============================================================================
# Train
# ============================================================================
.PHONY: train-local train-azure
train-local:  ## Train locally with mim (select backbone with EXP=1a..5a)
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

train-azure:  ## Submit an Azure ML training job (EXP=1a..5a; needs .env + `az login`)
	az ml job create \
		--file $(AZURE_TRAIN) \
		--subscription $(AZ_SUBSCRIPTION) \
		--resource-group $(AZ_RESOURCE_GROUP) \
		--workspace-name $(AZ_WORKSPACE)

# ============================================================================
# Test / inference
# ============================================================================
.PHONY: test-local test-flowity weather

# Path to the trained checkpoint to evaluate (override on the CLI).
CHECKPOINT_PATH ?= data/checkpoints/swin/seed_91/best_mIoU_iter_34000.pth
test-local:  ## Test a local checkpoint with mim (EXP=1a..5a; pass CHECKPOINT_PATH=...)
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

# Shared knobs for the two inference runners (see scripts/run/).
MODEL  ?= swin
MODE   ?= smoke
DEVICE ?= auto
SEED   ?=
test-flowity:  ## §4.1 Flowity test-set inference (MODEL=swin|flash|hrnet|beit|interimage MODE=smoke|full)
	bash scripts/run/run_flowity_test.sh $(MODEL) $(MODE) $(SEED) $(DEVICE)

weather:  ## §4.2 adverse-weather inference (MODEL=... MODE=smoke|full DEVICE=auto|gpu|cpu [SEED=])
	bash scripts/run/run_weather.sh $(MODEL) $(MODE) $(DEVICE) $(SEED)

# ============================================================================
# Azure job download / analysis / dev
# ============================================================================
.PHONY: download-jobs download-job-azure parse-logs num-params jupyter

download-jobs:  ## Download the thesis training jobs from Azure (bajar_jobs.sh; needs .env + `az login`)
	bash scripts/run/bajar_jobs.sh

JOB_ID ?=
download-job-azure:  ## Download+flatten one Azure PIPELINE parent job (JOB_ID=<parent-job-name>)
	python3 scripts/azure/download_job.py --job-id $(JOB_ID) --output-dir data/checkpoints

parse-logs:  ## Regenerate training/testing analytics xlsx from the downloaded logs
	python3 scripts/logs/parse_training_logs.py
	python3 scripts/logs/parse_test_logs.py

# Params + backbone GFLOPs. Defaults to the selected EXP's config; pass a
# CHECKPOINT=... for exact numbers. Runs inside road_defect_base.
CONFIG     ?= $(CONFIG_FILE)
CHECKPOINT ?=
num-params:  ## Model params + backbone GFLOPs (CONFIG=... [CHECKPOINT=...] or EXP=1a..5a)
	docker run --rm -v $(PWD):/app -w /app road_defect_base:latest \
		python3 scripts/analysis/model_complexity_measurement.py $(CONFIG) $(CHECKPOINT)

jupyter:  ## Launch Jupyter Lab in road_defect_base on http://localhost:8888
	docker run --rm -p 8888:8888 -v $(PWD):/app -w /app road_defect_base:latest \
		jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''
