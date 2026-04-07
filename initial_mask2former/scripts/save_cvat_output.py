import os
import shutil
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Map CVAT Cityscapes masks back to original image folders.")
    parser.add_argument("--cvat-dir", type=str, required=True, 
                        help="Path to the extracted CVAT export folder (contains gtFine files)")
    parser.add_argument("--work-dir", type=str, required=True, 
                        help="Path to the base directory containing all the job folders")
    return parser.parse_args()

def build_image_map(work_dir):
    """
    Scans the working directory to find all original images and maps their 
    base filenames (e.g., '1ea05d..._000000') to their exact folder paths.
    """
    print(f"Scanning '{work_dir}' for original image files...")
    img_map = {}
    valid_exts = ('.jpg', '.jpeg', '.png')
    
    for root, _, files in os.walk(work_dir):
        for f in files:
            if f.lower().endswith(valid_exts):
                # Extract just the file name without .jpg: '1ea05d7c90ce447c85ed0cfd81e92a74_000000'
                base_name = os.path.splitext(f)[0]
                img_map[base_name] = root
                
    print(f"Found {len(img_map)} original images to act as matching keys.")
    return img_map

def distribute_masks(cvat_dir, img_map):
    """
    Scans the CVAT directory, matches files to original images based on the filename, 
    and copies them to categorized folders back in the job directory.
    """
    print(f"\nScanning CVAT directory: '{cvat_dir}'...")
    
    # Sort image keys by length descending to ensure we match the longest possible name
    base_names = sorted(img_map.keys(), key=len, reverse=True)
    
    processed_count = 0
    unmatched_files = []

    for root, _, files in os.walk(cvat_dir):
        for f in files:
            if "gtFine" not in f:
                continue # Skip non-mask files
                
            # Determine mask type so we can put them in separate folders
            if "color" in f:
                mask_type = "color"
            elif "instanceIds" in f:
                mask_type = "instanceIds"
            elif "labelIds" in f:
                mask_type = "labelIds"
            else:
                continue

            # Find which original image filename this mask belongs to
            matched_base = None
            for base in base_names:
                if f.startswith(base):
                    matched_base = base
                    break
            
            if matched_base:
                # img_map[matched_base] is: .../clever_train_6k77vt5d01/dataloader_output/images
                img_folder = img_map[matched_base]
                
                # Go up two levels to reach the main job folder (.../clever_train_6k77vt5d01)
                dataloader_folder = os.path.dirname(img_folder)
                job_folder = os.path.dirname(dataloader_folder)
                
                # Create destination: .../clever_train_6k77vt5d01/cvat_output/color/
                dest_dir = os.path.join(job_folder, "cvat_output", mask_type)
                os.makedirs(dest_dir, exist_ok=True)
                
                src_path = os.path.join(root, f)
                
                # Rename the file back to a clean version to match the original image
                # Result: '1ea05d7c90ce447c85ed0cfd81e92a74_000000_color.png'
                clean_filename = f"{matched_base}_{mask_type}.png"
                dest_path = os.path.join(dest_dir, clean_filename)
                
                shutil.copy2(src_path, dest_path)
                processed_count += 1
            else:
                unmatched_files.append(f)

    print(f"\n--- Distribution Complete ---")
    print(f"Successfully mapped and copied {processed_count} mask files.")
    
    if unmatched_files:
        print(f"WARNING: Could not find original images for {len(unmatched_files)} CVAT files.")

if __name__ == "__main__":
    args = get_args()
    image_mapping = build_image_map(args.work_dir)
    
    if not image_mapping:
        print("No images found. Please check your --work-dir path.")
    else:
        distribute_masks(args.cvat_dir, image_mapping)