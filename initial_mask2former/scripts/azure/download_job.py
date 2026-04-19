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
        job_ids = ["helpful_duck_q8pgsrj0sb", "calm_battery_xgwbt99hg9", "quirky_wing_gqy29j47d1", "cyan_plow_hx0hqbf5fd", "magenta_machine_wgmjmg1w22", 
                   "purple_planet_vr129q38ql", "placid_beach_t6c182fgyh", "zen_rice_94w4vl9xbk", "ivory_fowl_1254x87q71", "joyful_carrot_6cqs202rpf",
                   "tough_zebra_qrqv939gm5", "icy_boniato_b5yh1949kl", "sharp_planet_w8x0fgt2cm", "olive_date_f6x0p43gk4", "mango_cat_49w5kztsn4",
                   "willing_cloud_t0w6bbgzyd", "jovial_muscle_xzdrbvczd1", "helpful_duck_q8pgsrj0sb", "amiable_date_c6ygz8ydg2"]
    
    target_components = ["dataloader",  "segmentation"]
    # Other options: "masker", "trafficsigns", "reconstruction"
    
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