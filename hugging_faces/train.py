# train.py
import torch
import logging
from transformers import logging as hf_logging
from transformers import (
    Mask2FormerForUniversalSegmentation, 
    Mask2FormerImageProcessor, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
    
)
from functools import partial
from dataset import BasicRoadDataset
from metrics import compute_metrics
import os
import glob
import wandb


hf_logging.set_verbosity_info()
logging.basicConfig(level=logging.INFO)

# 1. Setup Labels
id2label = {0: "bg", 1: "cracks", 2: "cracks_alligator", 3: "cracks_severe"}
label2id = {v: k for k, v in id2label.items()}

# 2. Initialize Processor and Model
# Using 512x512 as your base crop size
processor = Mask2FormerImageProcessor.from_pretrained(
    "facebook/mask2former-swin-tiny-ade-semantic",
    do_reduce_labels=False,
    size={"height": 512, "width": 512}
)

model = Mask2FormerForUniversalSegmentation.from_pretrained(
    "facebook/mask2former-swin-tiny-ade-semantic",
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

compute_metrics_fn = partial(
    compute_metrics, 
    processor=processor, 
    id2label=id2label
)

# 3. Data Loading
DATA_ROOT = 'data/2026-01-19-defect_dataset'
LABEL_DIR = 'labels_cracks'
OUTPUT_DIR = './work_dirs/basic_round_4'
run_id = os.path.basename(OUTPUT_DIR)

wandb.init(
    project="huggingface", 
    id=run_id, 
    resume="allow" # Automatically resumes if ID exists, starts new if it doesn't
)

train_dataset = BasicRoadDataset(DATA_ROOT, LABEL_DIR, 'train.txt', processor)
val_dataset = BasicRoadDataset(DATA_ROOT, LABEL_DIR, 'val.txt', processor)

# 4. Mandatory Mask2Former Collate Function
# Mask2Former requires mask_labels and class_labels to be lists, not stacked tensors
def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    pixel_mask = torch.stack([item["pixel_mask"] for item in batch])
    mask_labels = [item["mask_labels"] for item in batch]
    class_labels = [item["class_labels"].clone().detach().to(torch.long) for item in batch]
    
    # This is the "Label" the Trainer passes to compute_metrics
    labels = torch.stack([item["labels"] for item in batch]) 
    
    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "mask_labels": mask_labels,
        "class_labels": class_labels,
        "labels": labels 
    }

# 5. Training Args
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=10,                  # Short run to verify everything works
    per_device_train_batch_size=4,   # Safe for RTX 4070
    gradient_accumulation_steps=2,
    
    #eval_do_concat_batches=False,
    per_device_eval_batch_size=4,
    dataloader_num_workers=4,
    eval_accumulation_steps=1,
    fp16=True,
    #gradient_checkpointing=True,
    
    eval_strategy="epoch",
    #eval_steps=500,
    save_strategy="epoch",
    #save_steps=500,
    logging_steps=50,
    save_total_limit=3,
    #learning_rate=5e-5,
    label_names=["labels"],
    remove_unused_columns=False,     # CRITICAL for Mask2Former
    metric_for_best_model="mean_iou",
    greater_is_better=True,
    push_to_hub=False,
    report_to="wandb",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics_fn,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

checkpoints = glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*"))

if checkpoints:
    # Sort by step number to find the most recent one
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    print(f"--> Found existing checkpoint: {latest_checkpoint}. Resuming...")
    trainer.train(resume_from_checkpoint=latest_checkpoint)
else:
    print("--> No checkpoints found. Starting training from scratch...")
    trainer.train()
