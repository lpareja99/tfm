import argparse
import os
import shutil
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

def get_args():
    parser = argparse.ArgumentParser(description="Download and flatten Azure ML pipeline component outputs.")
    parser.add_argument("--job-id", type=str, help="The Parent Job ID (e.g., tender_stomach_mhgc9vmn9p)")
    parser.add_argument("--output-dir", type=str, default="download_job", help="The local directory to save downloaded artifacts")
    parser.add_argument("--sub", type=str, default="2dcd4ebb-39e0-451f-9dcb-9a3ec70e0299", help="Subscription ID")
    parser.add_argument("--rg", type=str, default="rg-flowityanalytics-testing", help="Resource Group name")
    parser.add_argument("--ws", type=str, default="ml-analytics-testing", help="Workspace name")
    return parser.parse_args()

def lift_output_folders(ml_client, job_name, base_path):
    # Temporary landing spot
    temp_dir = os.path.join(base_path, f"temp_{job_name}")
    
    print(f"   Downloading artifacts for {job_name}...")
    ml_client.jobs.download(name=job_name, download_path=temp_dir, all=True)
    
    found_output = False
    for root, dirs, files in os.walk(temp_dir):
        if root.endswith("named-outputs"):
            for folder_name in dirs:
                source = os.path.join(root, folder_name)
                destination = os.path.join(base_path, folder_name)
                
                if os.path.exists(destination):
                    shutil.rmtree(destination)
                shutil.move(source, destination)
                print(f"   ✓ Extracted folder: {folder_name}")
                found_output = True

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    if not found_output:
        print(f"   ! No named-outputs found for {job_name}")

def find_and_process_recursive(ml_client, parent_name, targets, base_path):
    children = list(ml_client.jobs.list(parent_job_name=parent_name))
    for child in children:
        display_name = child.display_name.lower()
        match = next((t for t in targets if t in display_name), None)
        
        if match:
            print(f"\nTarget Match Found: {child.display_name}")
            full_job = ml_client.jobs.get(child.name)
            lift_output_folders(ml_client, full_job.name, base_path)
        elif child.type in ["pipeline", "base"]:
            find_and_process_recursive(ml_client, child.name, targets, base_path)

def main():
    args = get_args()
    
    if args.job_id:
        job_ids = [args.job_id]
    else:
        job_ids = ["purple_wing_zf84jm3h37", "patient_bulb_h67htrw5yd", "sweet_oyster_clzv48lhq5",
                "dreamy_night_pg8g25vb72", "green_prune_3jd5v4h1p2", "jolly_plate_w9vr36jgg4",
                "sincere_kitchen_2mgcv5w65d", "clever_train_6k77vt5d01", "good_yacht_gjxdvkhxvg",
                "sincere_kitchen_vlbtkqsf5f", "teal_atemoya_5twc5bh411", "maroon_cheese_7dych1zwxy",
                "green_curtain_vtgxw75n43", "cool_quince_65rt4g50qw", "gentle_quince_6dmnf317ls",
                "witty_office_xhwpm61k5h", "maroon_carrot_ylmtx2j0z4", "great_kumquat_5v4kbfr1jg",
                "ivory_nail_723gmdlh0t", "tough_seed_5hcjhgtfyh", "yellow_pear_z5kxgzqqk6",
                "ashy_okra_y2kjvzr1qk"]
    
    target_components = ["dataloader",  "segmentation"]
    
    print(f"Connecting to Workspace: {args.ws}...")
    # Logic: Initialize the client ONCE outside the loop to save time
    ml_client = MLClient(DefaultAzureCredential(), args.sub, args.rg, args.ws)
    
    for job_id in job_ids:
        #target_components = ["dataloader", "masker", "segmentation", "trafficsigns", "reconstruction"]
        base_download_path = os.path.join(args.output_dir, job_id)

        print(f"Starting extraction for Job: {job_id}")
        os.makedirs(base_download_path, exist_ok=True)
        
        try:
            find_and_process_recursive(ml_client, job_id, target_components, base_download_path)
            print(f"✓ Finished {job_id}. Outputs: {os.path.abspath(base_download_path)}")
        except Exception as e:
            print(f"✗ Error processing {job_id}: {str(e)}")
            
        print(f"\nAll done! Processed {len(job_ids)} jobs.")

if __name__ == "__main__":
    main()