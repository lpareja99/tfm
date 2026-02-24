import torch, cv2, numpy as np, os, argparse, matplotlib.pyplot as plt
from pathlib import Path
from mmseg.apis import init_model
from mmengine.config import Config
from sklearn.manifold import TSNE

def get_patches_with_ids(img_p, mask_p, size=128):
    img, mask = cv2.imread(str(img_p)), cv2.imread(str(mask_p), 0)
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for c in cnts:
        if cv2.contourArea(c) < 100: continue
        x, y, w, h = cv2.boundingRect(c)
        cx, cy = x + w//2, y + h//2
        y1, y2 = max(0, cy - size//2), min(img.shape[0], cy + size//2)
        x1, x2 = max(0, cx - size//2), min(img.shape[1], cx + size//2)
        p, m_patch = img[y1:y2, x1:x2], mask[y1:y2, x1:x2]
        if p.shape[:2] == (size, size):
            valid_pixels = m_patch[m_patch > 0]
            if len(valid_pixels) > 0:
                class_id = np.bincount(valid_pixels).argmax()
                results.append({'img': cv2.resize(p, (224, 224)), 'id': class_id})
    return results

def get_feats(model, patches):
    model.eval()
    feats = []
    with torch.no_grad():
        for p in patches:
            t = torch.from_numpy(p['img']).permute(2, 0, 1).float().unsqueeze(0).cuda()
            t = (t - 127.5) / 127.5
            f = model.backbone(t)[-1]
            feats.append(torch.mean(f, dim=(2, 3)).cpu().numpy().flatten())
    return np.array(feats)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config'); parser.add_argument('checkpoint')
    parser.add_argument('data'); parser.add_argument('out')
    args = parser.parse_args()
    
    # DYNAMIC CONFIG LOADING
    cfg = Config.fromfile(args.config)
    class_names = cfg.class_names  # e.g., ('bg', 'cracks', ...) [cite: 4]
    palette = cfg.palette          # e.g., [[0,0,0], [250,50,83], ...] 
    
    model = init_model(args.config, args.checkpoint, device='cuda:0')
    img_dir, mask_dir = Path(args.data)/"images", Path(args.data)/"labels"
    
    data = []
    for ip in list(img_dir.glob("*.jpg"))[:50]:
        mp = mask_dir / (ip.stem + ".png")
        if mp.exists(): data.extend(get_patches_with_ids(ip, mp))
    
    if not data:
        print("No patches found."); exit()

    ids = [d['id'] for d in data]
    embeds = TSNE(n_components=2, init='pca').fit_transform(get_feats(model, data))
    
    plt.figure(figsize=(10, 8))
    for cid in np.unique(ids):
        mask = np.array(ids) == cid
        # Convert 0-255 RGB to 0-1 float for matplotlib
        color = np.array(palette[cid]) / 255.0 
        plt.scatter(embeds[mask, 0], embeds[mask, 1], alpha=0.7, 
                    c=[color], label=class_names[cid], edgecolors='w')
    
    plt.title(f"t-SNE Clustering: {Path(args.config).stem}")
    plt.legend()
    plt.savefig(Path(args.out)/"tsne_dynamic_clusters.png")