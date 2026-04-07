import os
import shutil
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Compile final test dataset from CVAT outputs.")
    parser.add_argument("--work-dir", type=str, required=True, 
                        help="Base directory with all job folders (e.g., data/laura_tfm_sun_22_dry)")
    parser.add_argument("--output-dir", type=str, required=True, 
                        help="Where to save the final test dataset")
    return parser.parse_args()

def build_test_dataset(work_dir, output_dir):
    images_out = os.path.join(output_dir, "images")
    labels_out = os.path.join(output_dir, "labels")
    test_txt_path = os.path.join(output_dir, "test.txt")

    # Create the final directory structure
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    valid_img_exts = ['.jpg', '.jpeg', '.png']
    processed_count = 0

    print(f"Scanning '{work_dir}' for completed annotations...")

    with open(test_txt_path, 'w') as txt_file:
        # Iterate through all the job folders in your working directory
        for job_folder in os.listdir(work_dir):
            job_path = os.path.join(work_dir, job_folder)
            if not os.path.isdir(job_path):
                continue
            
            # The paths we expect inside each job folder
            img_dir = os.path.join(job_path, "dataloader_output", "images")
            # Explicitly target the 1,2,3,4 masks (labelIds)
            label_dir = os.path.join(job_path, "cvat_output", "labelIds") 

            # Skip folders that don't have CVAT labelIds yet
            if not os.path.exists(img_dir) or not os.path.exists(label_dir):
                continue 
            
            # For every mask we find, go grab its matching image
            for mask_file in os.listdir(label_dir):
                if not mask_file.endswith(".png"):
                    continue
                
                # Extract the original base name (e.g., '1ea05d..._000000')
                base_name = mask_file.replace("_labelIds.png", "")

                # Look for the original image
                img_src = None
                img_ext = None
                for ext in valid_img_exts:
                    potential_img = os.path.join(img_dir, base_name + ext)
                    if os.path.exists(potential_img):
                        img_src = potential_img
                        img_ext = ext
                        break
                
                if img_src:
                    # 1. Copy the Image
                    img_dest = os.path.join(images_out, base_name + img_ext)
                    shutil.copy2(img_src, img_dest)

                    # 2. Copy the Mask (and rename it back to just base_name.png for mmsegmentation)
                    mask_src = os.path.join(label_dir, mask_file)
                    mask_dest = os.path.join(labels_out, base_name + ".png")
                    shutil.copy2(mask_src, mask_dest)

                    # 3. Add to test.txt
                    txt_file.write(f"{base_name}\n")
                    processed_count += 1
                else:
                    print(f"  Warning: Found mask {mask_file} but no matching image in {img_dir}")

    print(f"\n--- Dataset Compilation Complete ---")
    print(f"Successfully aggregated {processed_count} image/label pairs.")
    print(f"Images saved to: {images_out}")
    print(f"Labels saved to: {labels_out}")
    print(f"Text list saved to: {test_txt_path}")

if __name__ == "__main__":
    args = get_args()
    build_test_dataset(args.work_dir, args.output_dir)