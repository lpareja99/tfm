import argparse
import os
import shutil
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

def get_args():
    parser = argparse.ArgumentParser(description="Download and flatten Azure ML pipeline component outputs.")
    parser.add_argument("--job-id", type=str, help="The Parent Job ID (e.g., tender_stomach_mhgc9vmn9p)")
    parser.add_argument("--output-dir", type=str, default="download_job", help="The local directory to save downloaded artifacts")
    parser.add_argument("--sub", type=str, default=os.environ.get("AZ_SUBSCRIPTION"), help="Subscription ID (or set AZ_SUBSCRIPTION)")
    parser.add_argument("--rg", type=str, default=os.environ.get("AZ_RESOURCE_GROUP"), help="Resource Group name (or set AZ_RESOURCE_GROUP)")
    parser.add_argument("--ws", type=str, default=os.environ.get("AZ_WORKSPACE"), help="Workspace name (or set AZ_WORKSPACE)")
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
    
    if not args.job_id:
        parser_error = "Pass a parent job id with --job-id (e.g. --job-id tender_stomach_mhgc9vmn9p)"
        raise SystemExit(parser_error)
    job_ids = [args.job_id]

    # Pipeline components whose named-outputs we want to lift out. Adjust to the
    # components your parent pipeline job exposes.
    target_components = ["dataloader", "segmentation"]
    
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