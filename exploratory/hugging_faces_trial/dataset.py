import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class BasicRoadDataset(Dataset):
    def __init__(self, data_root, label_dir, split_file, processor, transform=None):
        self.data_root = data_root
        self.label_dir = label_dir
        self.processor = processor
        self.transform = transform
        
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
        
        image_np = np.array(Image.open(img_path).convert("RGB"))
        mask_np = np.array(Image.open(mask_path).convert("L"))
        
        if getattr(self, "transform", None) is not None:
            augmented = self.transform(image=image_np, mask=mask_np)
            image_np = augmented['image']
            mask_np = augmented['mask']
        
        inputs = self.processor(images=image_np, segmentation_maps=mask_np, return_tensors="pt")
        inputs = {k: v[0] for k, v in inputs.items()}
        inputs["labels"] = torch.tensor(mask_np, dtype=torch.long)
             
        return inputs
