EXP ?= 1a

var_1a = 1a_swin_t_cracks.env
var_1b = 1b_swin_t_all_defects.env
var_1c = 1c_swin_t_all_defects_relabel.env

var_2a = 2a_hrnet_t_all_defects.env
var_3a = 3a_intern_image.env
var_4a = 4a_flash_intern_image.env
var_5a = 5a_beit2_base.env

include runs/$(var_$(EXP))

# Azure/ACR coordinates come from a gitignored .env (see .env.example).
# `-include` = do not fail if .env is absent (e.g. on a fresh clone).
-include .env

# =============
# general
# =============
jobs = ["purple_wing_zf84jm3h37", "patient_bulb_h67htrw5yd"]
jobs2 = zen_rice_94w4vl9xbk
output_dir ?= "./data/laura_tfm_sun_22_dry_flowity_pipeline/"
input_dir ?= "./data/laura_tfm_dry_annotated/test_annotation_output"
tfm_data_output ?= "./data/final_test_dataset/"

download-job-flowity-pipeline:
	python3 scripts/azure/download_job.py --job-id $(jobs) --output-dir $(output_dir)

cvat_input_annotation_cration:
	python3 scripts/data_prep/mask_to_cvat.py --input-dir $(jobs2)

cvat_output_organization_mapping:
	python3 scripts/data_prep/save_cvat_output.py --cvat-dir $(input_dir) --work-dir $(output_dir)

build_test_dataset:
	python3 scripts/data_prep/build_test_dataset_cvat.py --work-dir $(output_dir) --output-dir $(tfm_data_output)

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
		--subscription $(AZ_SUBSCRIPTION) \
		--resource-group $(AZ_RESOURCE_GROUP) \
		--workspace-name $(AZ_WORKSPACE)


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
	python exploratory/initial_mask2former/scripts/mim_test_executer.py \
		--work-dir $(WORK_DIR) \
		--data-root $(LOCAL_DATA) \
		--label-dir $(LABEL)


# ==========================================
# test-flowity (§4.1) — inference on the FLOWITY test set
#   Historical command used to evaluate every model on Flowity data.
#   (Adverse-weather robustness inference §4.2 is the `weather` target below.)
#   usage: make test-flowity FLOWITY_CONFIG=config.py \
#          FLOWITY_CHECKPOINT=output/best_mIoU_iter_XXXX.pth FLOWITY_WORK_DIR=output
# ==========================================
FLOWITY_CONFIG ?= config.py
FLOWITY_CHECKPOINT ?= output/best_mIoU_iter_28000_flash.pth
FLOWITY_WORK_DIR ?= output
test-flowity:
	mim test mmseg $(FLOWITY_CONFIG) \
		--checkpoint $(FLOWITY_CHECKPOINT) \
		--work-dir $(FLOWITY_WORK_DIR) \
		--show-dir $(FLOWITY_WORK_DIR)/test_visuals \
		--cfg-options \
		test_dataloader.dataset.data_prefix.seg_map_path=labels_basic_defects_relabel

num_params_millions:
	python -c "
	from mmengine.config import Config
	from mmseg.models import build_segmentor
	cfg = Config.fromfile('config_tiny.py')
	model = build_segmentor(cfg.model)
	params = sum(p.numel() for p in model.parameters())
	print(f'Total Parameters: {params / 1e6:.2f} M')
	"

predictions:
	mim run mmseg test config.py output/best_mIoU_iter_17000_swin_t.pth --out output/predictions.pkl


config = config.py
checkpoint = output/best_mIoU_iter_17000_swin_t.pth
work_dir = ./output

benchmark:
	mim run mmseg benchmark {config} {checkpoint} --work-dir {work_dir}

	mim run mmseg benchmark config.py output/best_mIoU_iter_10000_intern_t.pth --work-dir ./output

seeds = 42, 1337, 2026, 777, 91

download_job_azure:
	python3 scripts/azure/download_job.py \
  		--job-id <NOMBRE_JOB_PADRE> \
  	    --output-dir data/checkpoints


run_jupyter:
	docker run --rm -p 8888:8888 \
  		-v /home/lpa/Documentos/tfm:/app -w /app \
  		road_defect_base:latest \
  		jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''

# ==========================================
# weather (§4.2) — adverse-weather robustness inference (all 5 models)
#   Single runner scripts/run/run_weather.sh; it picks image/config/checkpoint and device per
#   model (flash/interimage require GPU; swin/hrnet/beit run on CPU or GPU).
#   usage: make weather MODEL=swin MODE=smoke
#          make weather MODEL=flash MODE=full
#   SEED is OPTIONAL (defaults to each model's best seed); pass it only to
#   override and try another seed:  make weather MODEL=swin MODE=full SEED=42
#   outputs: data/output/weather/<MODEL>/<cond>/{pred_masks,vis}
# ==========================================
MODEL ?= swin
MODE ?= smoke
DEVICE ?= auto
SEED ?=
weather:
	bash scripts/run/run_weather.sh $(MODEL) $(MODE) $(DEVICE) $(SEED)