from pathlib import Path
import torch
import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import evaluate
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
from configs.config import DATA_ROOT, LABEL_DIR, TEST_SPLIT, ID2LABEL, BASE_MODEL, WORK_DIR

def get_image_list(data_root, split_file):
    split_path = Path(data_root) / 'splits' / split_file
    with open(split_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def inference_masks(image_names, checkpoint, data_root, masks_dir, device):
    
    print(f"--> Loading model from  {checkpoint}...")
    processor = Mask2FormerImageProcessor.from_pretrained(
        BASE_MODEL, do_reduce_labels=False, size={"height": 512, "width": 512}
    )
    #processor = Mask2FormerImageProcessor.from_pretrained(CHECKPOINT)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(checkpoint).to(device).eval() 
        
    with torch.no_grad():
        for img_name in tqdm(image_names):
            mask_path = masks_dir / f"{img_name}.png"
            if mask_path.exists(): continue
            
            img_path = Path(data_root) / 'images' / f"{img_name}.jpg"
            image = Image.open(img_path).convert("RGB")
            
            # Inference
            inputs = processor(images=image, return_tensors="pt").to(device)
            outputs = model(**inputs)
            
            prediction = processor.post_process_semantic_segmentation(
                outputs, target_sizes=[image.size[::-1]]
            )[0].cpu().numpy()
            
            Image.fromarray(prediction.astype(np.uint8)).save(mask_path)


def calculate_metrics(image_names, data_root, label_dir, masks_dir, id2label):
    
    print(" Metrics Calculation ...")
    metric = evaluate.load("mean_iou")
        
    for img_name in tqdm(image_names, desc="Evaluating"):
        gt_path = Path(data_root) / label_dir / f"{img_name}.png"
        pred_mask_path = masks_dir / f"{img_name}.png"
        
        gt_array = np.array(Image.open(gt_path).convert("L"), dtype=np.int32)
        pred_array = np.array(Image.open(pred_mask_path), dtype=np.int32)
        metric.add_batch(predictions=[pred_array], references=[gt_array])

    results = metric.compute(num_labels=len(id2label), ignore_index=255, reduce_labels=False)

    class_data = []
    for cid, label in id2label.items():
        iou = results["per_category_iou"][cid]
        acc = results["per_category_accuracy"][cid]
        class_data.append({
            "Label": label, "IoU": iou, "Recall": acc, 
            "Dice": (2 * iou) / (1 + iou) if not np.isnan(iou) else np.nan
        })
    
    return pd.DataFrame(class_data), results
    

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    work_path = Path(WORK_DIR)
    
    res_path = work_path / "results"
    masks_dir, analysis_dir = res_path / "masks", res_path / "analysis"
    for d in [masks_dir, analysis_dir]: d.mkdir(parents=True, exist_ok=True)

    checkpoints = list(work_path.glob("checkpoint-*"))
    if not checkpoints: raise FileNotFoundError("No se encontraron checkpoints")
    last_ckpt = max(checkpoints, key=lambda p: p.stat().st_ctime)
    
    image_names = get_image_list(DATA_ROOT, TEST_SPLIT)
    
    wandb.init(project="huggingface", id=work_path.name, resume="allow")

    inference_masks(image_names, last_ckpt, DATA_ROOT, masks_dir, device)
    df_class, raw_results = calculate_metrics(image_names, DATA_ROOT, LABEL_DIR, masks_dir, ID2LABEL)
    df_class.to_csv(analysis_dir / "class_metrics.csv", index=False)
    df_global = pd.DataFrame({
        "Metric": ["Mean IoU", "Mean Accuracy", "Overall Accuracy"],
        "Score": [raw_results['mean_iou'], raw_results['mean_accuracy'], raw_results['overall_accuracy']]
    })
    df_global.to_csv(analysis_dir / "global_metrics.csv", index=False)
    
    wandb.log({
        "test/Mean_IoU": raw_results['mean_iou'],
        "test/Overall_Acc": raw_results['overall_accuracy'],
        "test/class_table": wandb.Table(dataframe=df_class)
    })
    
    # Plot simple
    df_class.dropna().plot(kind='bar', x='Label', figsize=(10,6), grid=True)
    plt.title("Metrics per Class")
    plt.tight_layout()
    plt.savefig(analysis_dir / "plot.png")
    wandb.log({"test/plot": wandb.Image(str(analysis_dir / "plot.png"))})
    
    wandb.finish()
    print("Done!")

    print("Test process Completed")


if __name__ == "__main__":
    main()