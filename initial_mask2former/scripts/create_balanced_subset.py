import os, shutil, random, numpy as np
from PIL import Image
from collections import defaultdict

# --- CONFIGURATION ---
BASE_DIR = '/app/data'
IMG_DIR= f'{BASE_DIR}/2026-01-19-defect_dataset/images'
LBL_DIR = f'{BASE_DIR}/2026-01-19-defect_dataset/labels'
OUT_IMG = f'{BASE_DIR}/subset_500/images'
OUT_LBL = f'{BASE_DIR}/subset_500/labels'
TARGET_TOTAL = 500
MIN_PER_CLASS = 20

def create_balanced_subset():
    # 1. Scan and Filter Rare Classes
    class_to_images = defaultdict(list)
    image_to_classes = defaultdict(set)
    label_files = [f for f in os.listdir(LBL_DIR) if f.endswith('.png')]

    for f in label_files:
        mask = np.array(Image.open(os.path.join(LBL_DIR, f)).convert('L'))
        for cls in np.unique(mask):
            if cls != 0:
                class_to_images[cls].append(f)
                image_to_classes[f].add(cls)

    # Remove classes that don't meet the minimum frequency threshold
    valid_classes = {c for c, imgs in class_to_images.items() if len(imgs) >= MIN_PER_CLASS}
    
    # Re-clean the image mapping to only include valid classes
    clean_image_to_classes = {f: (classes & valid_classes) for f, classes in image_to_classes.items() if (classes & valid_classes)}

    # 2. Greedy Selection for Uniform Distribution
    selected = set()
    current_counts = defaultdict(int)
    target_per_class = TARGET_TOTAL // len(valid_classes)


    while len(selected) < TARGET_TOTAL and clean_image_to_classes:
        # Score candidates: prioritize images containing classes we are furthest from target
        best_f = max(clean_image_to_classes.keys(), 
                     key=lambda f: sum(target_per_class - current_counts[c] for c in clean_image_to_classes[f]))
        
        selected.add(best_f)
        for c in clean_image_to_classes[best_f]: current_counts[c] += 1
        del clean_image_to_classes[best_f] # Remove from pool once picked

    # 3. Export with Extension Swapping (.png label -> .jpg image)
    os.makedirs(OUT_IMG, exist_ok=True); os.makedirs(OUT_LBL, exist_ok=True)
    for f in selected:
        img_name = f.replace('.png', '.jpg')
        if os.path.exists(os.path.join(IMG_DIR, img_name)):
            shutil.copy(os.path.join(IMG_DIR, img_name), os.path.join(OUT_IMG, img_name))
            shutil.copy(os.path.join(LBL_DIR, f), os.path.join(OUT_LBL, f))

    print(f"✅ Created subset of {len(selected)} images in {OUT_IMG}")
    
    return selected, image_to_classes
    

def create_train_val_split(selected_images, image_to_classes):

    # 1. Setup paths
    splits_dir = os.path.join('/app/data/subset_500', 'splits')
    os.makedirs(splits_dir, exist_ok=True)

    # 2. Preparation
    # We use the set of images you already selected in the previous step
    all_selected_labels = list(selected_images) 
    val_size = int(len(all_selected_labels) * 0.2) # 20% for validation (100 images)
    val_names = set()
    val_counts = defaultdict(int)

    # 3. Greedy Selection for Validation Set
    # We pick the 100 images for val.txt that best represent all classes
    pool = {f: image_to_classes[f] for f in all_selected_labels}

    print(f"Creating balanced splits for {len(all_selected_labels)} images...")

    while len(val_names) < val_size and pool:
        # Score images: Pick the one that helps the most under-represented classes in Val
        best_f = max(pool.keys(), key=lambda f: sum(1 for c in pool[f] if val_counts[c] == 0))
        
        val_names.add(best_f)
        for c in pool[best_f]:
            val_counts[c] += 1
        del pool[best_f]

    # 4. Separate names and strip extensions (as seen in your train.txt example)
    # The remaining images in the pool go to training
    train_names = [os.path.splitext(f)[0] for f in all_selected_labels if f not in val_names]
    val_names_final = [os.path.splitext(f)[0] for f in val_names]

    # 5. Write to files
    with open(os.path.join(splits_dir, 'train.txt'), 'w') as f:
        f.write('\n'.join(train_names))

    with open(os.path.join(splits_dir, 'val.txt'), 'w') as f:
        f.write('\n'.join(val_names_final))

    print(f"✅ Balanced split files created in {splits_dir}")
    print(f"Stats: {len(train_names)} train images, {len(val_names_final)} val images.")

if __name__ == "__main__":
    selected_images, image_to_classes = create_balanced_subset()
    create_train_val_split(selected_images, image_to_classes)