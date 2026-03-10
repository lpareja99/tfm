# train.py
import torch
import logging
from transformers import logging as hf_logging
from transformers import (
    Mask2FormerForUniversalSegmentation, 
    Mask2FormerImageProcessor, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback,
)
from functools import partial
from dataset import BasicRoadDataset
from metrics import compute_metrics
import os
import glob
import wandb
import albumentations as A
from configs.config import DATA_ROOT, LABEL_DIR, TRAIN_SPLIT, VAL_SPLIT, ID2LABEL, BASE_MODEL, WORK_DIR


#hf_logging.set_verbosity_info()
#logging.basicConfig(level=logging.INFO)


# 2. Initialize Processor and Model
processor = Mask2FormerImageProcessor.from_pretrained(
    BASE_MODEL,
    do_reduce_labels=False,
    size={"height": 512, "width": 512}
)

model = Mask2FormerForUniversalSegmentation.from_pretrained(
    BASE_MODEL,
    id2label=ID2LABEL,
    label2id={v: k for k, v in ID2LABEL.items()},
    ignore_mismatched_sizes=True
)

compute_metrics_fn = partial(
    compute_metrics, 
    processor=processor, 
    id2label=ID2LABEL
)

run_id = os.path.basename(WORK_DIR)

wandb.init(
    project="huggingface", 
    id=run_id, 
    resume="allow" # Automatically resumes if ID exists, starts new if it doesn't
)

train_transform = A.Compose([
    A.RandomScale(scale_limit=(-0.5, 1.0), p=1.0), 
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.ToGray(p=0.1),
    # PadIfNeeded prevents errors if RandomScale makes the image too small to crop
    A.PadIfNeeded(min_height=512, min_width=512, border_mode=0, fill_mask=255),
    A.RandomCrop(height=512, width=512),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Resize(height=512, width=512)
])

# Validation should NOT be distorted or cropped randomly. It just needs to be the right size.
val_transform = A.Compose([
    A.Resize(height=512, width=512)
])

train_dataset = BasicRoadDataset(DATA_ROOT, LABEL_DIR, TRAIN_SPLIT, processor, transform=train_transform)
val_dataset = BasicRoadDataset(DATA_ROOT, LABEL_DIR, VAL_SPLIT, processor, transform=val_transform)

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
    output_dir=WORK_DIR,
    num_train_epochs=2,                  # Short run to verify everything works
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

checkpoints = glob.glob(os.path.join(WORK_DIR, "checkpoint-*"))

processor.save_pretrained(WORK_DIR)

if checkpoints:
    # Sort by step number to find the most recent one
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    print(f"--> Found existing checkpoint: {latest_checkpoint}. Resuming...")
    trainer.train(resume_from_checkpoint=latest_checkpoint)
else:
    print("--> No checkpoints found. Starting training from scratch...")
    trainer.train()
