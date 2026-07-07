import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import matplotlib.patches as mpatches

# ==============================================================================
# 1. CONFIGURACIÓN DE RUTAS
# ==============================================================================
BASE_DIR = Path("data/2026-01-19-defect_dataset")
DIR_RGB = BASE_DIR / "images"
DIR_GT_REMAPPED = BASE_DIR / "labels_test_ids_ready"
DIR_GT_RELABEL = BASE_DIR / "labels_basic_defects_relabel"
DIR_PRED_SWIN = Path("data/output/swin/eval_results")
DIR_PRED_FLASH = Path("data/output/flash/eval_results")
SPLIT_FILE = BASE_DIR / "splits" / "test_ready.txt"

DIR_OUTPUT_VIS = Path("data/output/results/side2side_full_frame_analysis")
MAX_IMAGES = 20 

# ==============================================================================
# 2. PALETAS Y CLASES
# ==============================================================================
PALETTE = [[0,0,0], [250,50,83], [36,179,83], [102,204,255], [255,165,0], [128,128,128], [255,255,0], [0,255,255], [255,0,255]]
CLASS_NAMES = ("bg", "cracks", "cracks_alligator", "cracks_severe", "edge_cracks", "fretting", "pothole", "manhole", "pole_shadow")

# ==============================================================================
# 3. FUNCIONES ANALÍTICAS
# ==============================================================================
def calculate_iou(pred, gt, num_classes=9):
    ious = []
    for c in range(1, num_classes):
        p = (pred == c)
        g = (gt == c)
        union = np.logical_or(p, g).sum()
        if union == 0: continue
        intersection = np.logical_and(p, g).sum()
        ious.append(intersection / union)
    return np.mean(ious) if ious else 0.0

def create_error_map(pred_ids, gt_ids):
    """Verde=Acierto, Rojo=Omitido(FN), Azul=Falso Positivo(FP)"""
    h, w = gt_ids.shape
    error_map = np.zeros((h, w, 3), dtype=np.uint8)
    tp = (pred_ids > 0) & (pred_ids == gt_ids)
    fn = (gt_ids > 0) & (pred_ids == 0)
    fp = (pred_ids > 0) & (pred_ids != gt_ids)
    error_map[tp] = [0, 255, 0]   # Verde
    error_map[fn] = [255, 0, 0]   # Rojo
    error_map[fp] = [0, 0, 255]   # Azul
    return error_map

def create_overlay(bg, mask, alpha=0.4):
    if bg is None or mask is None: return mask
    if bg.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (bg.shape[1], bg.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.addWeighted(bg, alpha, mask, 1 - alpha, 0)

def apply_palette(path):
    try:
        mask = np.array(Image.open(path))
        rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for i, c in enumerate(PALETTE): rgb[mask == i] = c
        return rgb, mask
    except: return None, None

# ==============================================================================
# 4. EJECUCIÓN
# ==============================================================================
def main():
    DIR_OUTPUT_VIS.mkdir(parents=True, exist_ok=True)
    with open(SPLIT_FILE, 'r') as f:
        basenames = [Path(l.strip()).stem for l in f if l.strip()]
    
    random.shuffle(basenames)

    for i, base in enumerate(basenames[:MAX_IMAGES]):
        img_rgb = None
        for ext in ['.jpg', '.png', '.jpeg']:
            p = DIR_RGB / f"{base}{ext}"
            if p.exists():
                img_rgb = np.array(Image.open(p).convert('RGB'))
                break
        if img_rgb is None: continue

        # Cargar datos
        rgb_remap, ids_remap = apply_palette(DIR_GT_REMAPPED / f"{base}.png")
        rgb_relabel, ids_relabel = apply_palette(DIR_GT_RELABEL / f"{base}.png")
        rgb_swin, ids_swin = apply_palette(DIR_PRED_SWIN / f"{base}.png")
        rgb_flash, ids_flash = apply_palette(DIR_PRED_FLASH / f"{base}.png")

        # Usar Relabel como referencia para métricas
        ref_ids = ids_relabel if ids_relabel is not None else ids_remap
        
        # Generar Vistas
        ov_relabel = create_overlay(img_rgb, rgb_relabel)
        ov_swin = create_overlay(img_rgb, rgb_swin)
        ov_flash = create_overlay(img_rgb, rgb_flash)
        err_swin = create_overlay(img_rgb, create_error_map(ids_swin, ref_ids))
        err_flash = create_overlay(img_rgb, create_error_map(ids_flash, ref_ids))

        # Plot 4x2
        fig, axes = plt.subplots(4, 2, figsize=(20, 26))
        fig.suptitle(f"Full Frame Forensic Analysis: {base}", fontsize=28, weight='bold')

        def show(ax, img, title, metrics=None):
            if img is not None:
                ax.imshow(img)
                if metrics:
                    ax.text(20, 50, metrics, color='white', fontsize=20, 
                            bbox=dict(facecolor='black', alpha=0.7))
            ax.set_title(title, fontsize=18); ax.axis('off')

        # Fila 1: Origen y Referencia
        show(axes[0, 0], img_rgb, "1. Original RGB Image")
        show(axes[0, 1], ov_relabel, "2. GT Relabeled (Gold Standard Overlay)")
        
        # Fila 2: Calidad de Etiquetas
        show(axes[1, 0], ov_relabel, "3. Manual Relabel View")
        show(axes[1, 1], create_overlay(img_rgb, create_error_map(ids_remap, ids_relabel)), 
             "4. Label Differences: Auto-Remap vs Manual-Relabel")

        # Fila 3: Resultados Swin
        show(axes[2, 0], ov_swin, "5. Swin-T Prediction", f"mIoU: {calculate_iou(ids_swin, ref_ids):.3f}")
        show(axes[2, 1], err_swin, "6. Swin-T Error Map (G:TP, R:FN, B:FP)")

        # Fila 4: Resultados Flash
        show(axes[3, 0], ov_flash, "7. Flash InternImage Prediction", f"mIoU: {calculate_iou(ids_flash, ref_ids):.3f}")
        show(axes[3, 1], err_flash, "8. Flash Error Map (G:TP, R:FN, B:FP)")

        # Leyenda unificada al final
        patches = [mpatches.Patch(color='green', label='Acierto (TP)'), 
                   mpatches.Patch(color='red', label='Omitido (FN)'), 
                   mpatches.Patch(color='blue', label='Falso Positivo (FP)')]
        fig.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, 0.02), 
                   ncol=3, fontsize=16, title="Error Map Legend")

        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        plt.savefig(DIR_OUTPUT_VIS / f"forensic_full_{base}.jpg", dpi=100)
        plt.close()
        print(f"  [+] Reporte generado: {base}")

if __name__ == "__main__":
    main()