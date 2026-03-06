import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from transformers import Mask2FormerImageProcessor

class BasicRoadDataset(Dataset):
    def __init__(self, data_root, label_dir, split_file, processor):
        self.data_root = data_root
        self.label_dir = label_dir
        self.processor = processor
        
        # Load the split file
        split_path = os.path.join(data_root, 'splits', split_file)
        with open(split_path, 'r') as f:
            self.image_names = [line.strip() for line in f.readlines()]
            
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        
        # Paths based on your structure
        img_path = os.path.join(self.data_root, 'images', f"{img_name}.jpg")
        mask_path = os.path.join(self.data_root, self.label_dir, f"{img_name}.png")
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") # 0: bg, 1: cracks, etc.

        image_resized_for_eval = image.resize((512, 512), Image.BILINEAR)
        mask_resized_for_eval = mask.resize((512, 512), Image.NEAREST)
        # Basic Preprocessing
        # The processor handles resizing, rescaling, and normalization
        inputs = self.processor(image_resized_for_eval, segmentation_maps=mask_resized_for_eval, return_tensors="pt")
        inputs = {k: v[0] for k, v in inputs.items()} 
        inputs["labels"] = torch.tensor(np.array(mask_resized_for_eval), dtype=torch.long)
        
     
        return inputs

# Replicating your training pipeline
'''
train_transform = A.Compose([
    A.RandomScale(scale_limit=(0.5, 2.0)),
    A.RandomBrightnessContrast(p=0.5), # PhotoMetricDistortion
    A.ToGray(p=0.1),                   # RandomGrayscale
    A.RandomCrop(width=512, height=512), # crop_size
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5)              # RandomFlip
])

val_transform = A.Compose([
    A.Resize(height=512, width=2048)   # Keeping aspect ratio conceptually
])
'''