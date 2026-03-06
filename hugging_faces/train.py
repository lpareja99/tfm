# train.py
import torch
from transformers import (
    Mask2FormerForUniversalSegmentation, 
    Mask2FormerImageProcessor, 
    TrainingArguments, 
    Trainer
)
from dataset import BasicRoadDataset

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

# 3. Data Loading
DATA_ROOT = 'data/2026-01-19-defect_dataset'
LABEL_DIR = 'labels_cracks'
OUTPUT_DIR = './work_dirs/basic_round_2'

train_dataset = BasicRoadDataset(DATA_ROOT, LABEL_DIR, 'train.txt', processor)
val_dataset = BasicRoadDataset(DATA_ROOT, LABEL_DIR, 'val.txt', processor)

# 4. Mandatory Mask2Former Collate Function
# Mask2Former requires mask_labels and class_labels to be lists, not stacked tensors
def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    pixel_mask = torch.stack([item["pixel_mask"] for item in batch])
    
    mask_labels = [item["mask_labels"] for item in batch]
    class_labels = [torch.tensor(item["class_labels"], dtype=torch.long) for item in batch]
    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "mask_labels": mask_labels,
        "class_labels": class_labels
    }

# 5. Training Args
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    max_steps=2000,                  # Short run to verify everything works
    per_device_train_batch_size=2,   # Safe for RTX 4070
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_steps=50,
    save_total_limit=3,
    #learning_rate=5e-5,
    remove_unused_columns=False,     # CRITICAL for Mask2Former
    push_to_hub=False,
    report_to="tensorboard",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
)

trainer.train()