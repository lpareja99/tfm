import os
import random

# Path to your original split file
original_split = 'data/2026-01-19-defect_dataset/splits/test.txt'
# Path for the new subset file
subset_split = 'data/2026-01-19-defect_dataset/splits/val.txt'

# 1. Read all lines from the original file
if not os.path.exists(original_split):
    print(f"Error: {original_split} not found!")
else:
    with open(original_split, 'r') as f:
        lines = f.readlines()

    # 2. Clean the lines (remove empty whitespace/newlines)
    lines = [line.strip() for line in lines if line.strip()]

    # 3. Check if we have enough images
    num_to_sample = min(len(lines), 200)

    # 4. Randomly select 200 items (with a fixed seed for reproducibility)
    random.seed(42)
    subset_lines = random.sample(lines, num_to_sample)

    # 5. Write to the new file
    with open(subset_split, 'w') as f:
        f.write('\n'.join(subset_lines) + '\n')

    print(f"Successfully created {subset_split} with {num_to_sample} random images.")