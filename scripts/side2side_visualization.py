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

# Entradas
DIR_RGB = BASE_DIR / "images"
DIR_GT_NEW = BASE_DIR / "labels_test_colors"
DIR_GT_REMAPPED = BASE_DIR / "labels_test_ids_ready"
DIR_GT_RELABEL = BASE_DIR / "labels_basic_defects_relabel"
DIR_PRED_SWIN = Path("data/output/swin/eval_results")
DIR_PRED_FLASH = Path("data/output/flash/eval_results")
SPLIT_FILE = BASE_DIR / "splits" / "test_ready.txt"

# Salida
DIR_OUTPUT_VIS = Path("results/side2side_visualization")
MAX_IMAGES = 20  # Cambia a None para procesar todo el dataset

# ==============================================================================
# 2. PALETAS Y MAPEOS DE COLOR
# ==============================================================================
PALETTE = [
    [0, 0, 0],       # 0: bg
    [250, 50, 83],   # 1: cracks
    [36, 179, 83],   # 2: cracks_alligator
    [102, 204, 255], # 3: cracks_severe
    [255, 165, 0],   # 4: edge_cracks
    [128, 128, 128], # 5: fretting
    [255, 255, 0],   # 6: pothole
    [0, 255, 255],   # 7: manhole
    [255, 0, 255]    # 8: pole_shadow
]

CLASS_NAMES = ("bg", "cracks", "cracks_alligator", "cracks_severe", 
               "edge_cracks", "fretting", "pothole", "manhole", "pole_shadow")

# Mapeo de Flowity (Color Viejo -> Color Alineado al Modelo/Contraste)
FLOWITY_MAPPING = [
    # --- FONDO ---
    ("background",      [0, 0, 0],       [0, 0, 0]),         # Negro puro

    # --- GRUPO 1: DEFECTOS ESTRUCTURALES (Colores Vibrantes) ---
    ("crack",           [250, 50, 83],   [255, 0, 0]),       # Rojo Puro
    ("severe crack",    [102, 255, 102], [0, 255, 255]),     # Cian Eléctrico
    ("network crack",   [36, 179, 83],   [0, 255, 0]),       # Verde Neón
    ("edge breaks",     [255, 0, 255],   [255, 0, 255]),     # Magenta
    ("pothole",         [115, 51, 128],  [255, 255, 0]),     # Amarillo Brillante
    ("gravel_hole",     [40, 17, 104],   [255, 215, 0]),     # Oro (Distinto al amarillo)
    ("stone hole",      [112, 160, 132], [255, 165, 0]),     # Naranja Intenso

    # --- GRUPO 2: ELEMENTOS DE VÍA Y SOMBRAS (Escala de Grises/Fríos) ---
    ("manhole",         [34, 62, 209],   [0, 0, 255]),       # Azul Puro
    ("dropped manhole", [152, 51, 171],  [0, 100, 255]),     # Azul Real
    ("pole shadow",     [172, 84, 109],  [130, 130, 130]),   # Gris Medio
    ("fretting",        [204, 153, 51],  [180, 180, 180]),   # Gris Claro

    # --- GRUPO 3: MANTENIMIENTO Y REPARACIONES (Tonos Oscuros/Tierra) ---
    ("Damage Patches",  [48, 103, 159],  [128, 0, 0]),       # Granate
    ("large repair",    [129, 160, 68],  [0, 0, 128]),       # Azul Marino
    ("patched",         [63, 63, 63],    [128, 128, 0]),     # Oliva
    ("bitumen seal",    [255, 173, 153], [165, 42, 42]),     # Marrón Rojizo
    ("loose stones",    [128, 35, 140],  [210, 180, 140]),   # Canela/Tan

    # --- GRUPO 4: SUPERFICIE Y OTROS (Tonos Pastel/Inusuales) ---
    ("bad joint",       [224, 68, 45],   [255, 105, 180]),   # Rosa Hot
    ("joint",           [93, 212, 109],  [220, 190, 255]),   # Lavanda
    ("coarse surface",  [191, 241, 121], [75, 0, 130]),      # Índigo
    ("edge grass",      [213, 164, 25],  [0, 128, 0]),       # Verde Bosque
    ("general defect",  [12, 227, 33],   [255, 250, 205]),   # Crema
    ("sill",            [36, 223, 0],    [0, 255, 127]),     # Verde Primavera
    ("standing water",  [145, 82, 135],  [0, 191, 255]),     # Azul Cielo Deep
    ("wet area",        [133, 221, 236], [173, 216, 230])    # Azul Pálido
]

# ==============================================================================
# 3. FUNCIONES DE PROCESAMIENTO
# ==============================================================================
def create_overlay(background_rgb, mask_rgb, alpha=0.5):
    """Mezcla imagen y máscara. Redimensiona la máscara si es necesario."""
    if background_rgb is None or mask_rgb is None: 
        return mask_rgb
    if background_rgb.shape[:2] != mask_rgb.shape[:2]:
        mask_rgb = cv2.resize(mask_rgb, (background_rgb.shape[1], background_rgb.shape[0]), 
                              interpolation=cv2.INTER_NEAREST)
    return cv2.addWeighted(background_rgb, alpha, mask_rgb, 1 - alpha, 0)

def remap_flowity_colors(img_rgb):
    if img_rgb is None: return None
    new_img = np.zeros_like(img_rgb)
    for _, old_c, new_c in FLOWITY_MAPPING:
        mask = np.all(img_rgb == old_c, axis=-1)
        new_img[mask] = new_c
    return new_img

def apply_palette(mask_path):
    try:
        mask = np.array(Image.open(mask_path))
        h, w = mask.shape
        rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for class_id, color in enumerate(PALETTE):
            rgb_mask[mask == class_id] = color
        return rgb_mask
    except: return None

def load_rgb(image_path):
    try: return np.array(Image.open(image_path).convert('RGB'), dtype=np.uint8)
    except: return None

# ==============================================================================
# 4. EJECUCIÓN PRINCIPAL
# ==============================================================================
def main():
    DIR_OUTPUT_VIS.mkdir(parents=True, exist_ok=True)
    
    with open(SPLIT_FILE, 'r') as f:
        basenames = [Path(line.strip()).stem for line in f if line.strip()]
    
    random.shuffle(basenames)
    print(f"[*] Generando visualizaciones aleatorias (Límite: {MAX_IMAGES})...")

    for i, basename in enumerate(basenames[:MAX_IMAGES]):
        # Buscar Imagen RGB
        img_rgb = None
        for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']:
            p = DIR_RGB / f"{basename}{ext}"
            if p.exists():
                img_rgb = load_rgb(p)
                break
        if img_rgb is None: continue

        # Cargar Máscaras y Procesar
        img_gt_flowity = remap_flowity_colors(load_rgb(DIR_GT_NEW / f"{basename}_color.png"))
        img_gt_remap = apply_palette(DIR_GT_REMAPPED / f"{basename}.png")
        img_gt_relabel = apply_palette(DIR_GT_RELABEL / f"{basename}.png")
        img_swin = apply_palette(DIR_PRED_SWIN / f"{basename}.png")
        img_flash = apply_palette(DIR_PRED_FLASH / f"{basename}.png")

        # Crear Overlays
        ov_flowity = create_overlay(img_rgb, img_gt_flowity)
        ov_remap = create_overlay(img_rgb, img_gt_remap)
        ov_relabel = create_overlay(img_rgb, img_gt_relabel)
        ov_swin = create_overlay(img_rgb, img_swin)
        ov_flash = create_overlay(img_rgb, img_flash)

        # Configurar Figura 3x2
        fig, axes = plt.subplots(3, 2, figsize=(22, 25))
        fig.suptitle(f" Mask Analysis - {basename}", fontsize=26, weight='bold')

        def show(ax, img, title):
            if img is not None: ax.imshow(img)
            else: ax.text(0.5, 0.5, 'Sin Datos', ha='center', va='center')
            ax.set_title(title, fontsize=18); ax.axis('off')

        show(axes[0, 0], img_rgb, "1. Original RGB Image")
        show(axes[0, 1], ov_flowity, "2. New Flowity GT data 25 Classes")
        show(axes[1, 0], ov_remap, "3. New Flowity Data GT Remap to 9 classes")
        show(axes[1, 1], ov_relabel, "4. Original GT Model Data 9 Classes")
        show(axes[2, 0], ov_swin, "5. Inference: Swin-T")
        show(axes[2, 1], ov_flash, "6. Inference: Flash InternImage")

        # Leyendas
        patches_model = [mpatches.Patch(color=[c/255.0 for c in PALETTE[i]], label=CLASS_NAMES[i]) for i in range(len(CLASS_NAMES))]
        patches_flow = [mpatches.Patch(color=[c[2][0]/255.0, c[2][1]/255.0, c[2][2]/255.0], label=c[0]) for c in FLOWITY_MAPPING]

        fig.legend(handles=patches_model, loc='lower center', bbox_to_anchor=(0.25, 0.01), ncol=3, title="Modelo (9 Clases)", title_fontsize=18)
        fig.legend(handles=patches_flow, loc='lower center', bbox_to_anchor=(0.7, 0.01), ncol=4, title="Flowity (25 Clases)", title_fontsize=18)

        plt.tight_layout(rect=[0, 0.06, 1, 0.96])
        plt.savefig(DIR_OUTPUT_VIS / f"triple_panel_{basename}.jpg", dpi=130)
        plt.close()
        print(f"  [+] {i+1}/{len(basenames[:MAX_IMAGES])} completada.")

if __name__ == "__main__":
    main()