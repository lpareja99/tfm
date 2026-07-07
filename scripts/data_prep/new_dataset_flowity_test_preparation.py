import numpy as np
import shutil
from PIL import Image
from pathlib import Path

# --- 1. CONFIGURACIÓN ---
BASE_DIR = Path("data/2026-01-19-defect_dataset")
MASKS_EXTRACTED_DIR = Path("data/temp_masks_extracted")

SPLITS_FILE = BASE_DIR / "splits" / "test.txt"
NEW_SPLIT_FILE = BASE_DIR / "splits" / "test_ready.txt"

# Carpetas finales (ya listas para el modelo)
OUTPUT_IDS_DIR = BASE_DIR / "labels_test_ids_ready"
OUTPUT_COLORS_DIR = BASE_DIR / "labels_test_colors"

TARGET_SIZE = (1280, 640) # Tamaño para evitar el IndexError del dataloader

NUEVOS_IDS = {
    5: 1, 20: 3, 9: 5, 19: 6, 15: 7, 18: 8, # Directos
    16: 2, 7: 4, 6: 7,                      # Merges
    11: 6, 23: 6                            # Opcionales (baches)
}

def main():
    OUTPUT_IDS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_COLORS_DIR.mkdir(parents=True, exist_ok=True)

    # Crear tabla de búsqueda para remapeo ultra-rápido
    lookup_table = np.zeros(256, dtype=np.uint8)
    for new_id, old_id in NUEVOS_IDS.items():
        lookup_table[new_id] = old_id

    # Leer el test.txt original y guardar las líneas
    with open(SPLITS_FILE, 'r') as f:
        test_basenames = {Path(line.strip()).stem: line.strip() for line in f if line.strip()}
    
    imagenes_validas = set()

    print("[*] Filtrando, remapeando, redimensionando y renombrando en 1 paso...")
    
    # --- 2. PROCESAMIENTO EN BUCLE ÚNICO ---
    for mask_path in MASKS_EXTRACTED_DIR.rglob('*.png'):
        mask_name = mask_path.name
        
        for test_base in test_basenames:
            if mask_name.startswith(test_base + "_gtFine") or mask_name.startswith(test_base + "_"):
                
                # A. Procesar IDs (Para el modelo)
                if mask_name.endswith("_gtFine_labelIds.png"):
                    # 1. Leer y remapear
                    img_array = np.array(Image.open(mask_path))
                    remapped_array = lookup_table[img_array]
                    
                    # 2. Redimensionar (Vecino más cercano)
                    remapped_img = Image.fromarray(remapped_array)
                    final_img = remapped_img.resize(TARGET_SIZE, Image.Resampling.NEAREST)
                    
                    # 3. Guardar con el nombre limpio (ej: "image_92.png")
                    clean_name = f"{test_base}.png"
                    final_img.save(OUTPUT_IDS_DIR / clean_name)
                    
                    imagenes_validas.add(test_base) # Marcamos que esta imagen funcionó
                
                # B. Procesar Colores (Para visualización)
                elif mask_name.endswith("_gtFine_color.png"):
                    clean_name = f"{test_base}_color.png"
                    shutil.copy2(mask_path, OUTPUT_COLORS_DIR / clean_name)
                
                break # Máscara procesada, pasamos a la siguiente

    # --- 3. CREAR EL NUEVO SPLIT ---
    with open(NEW_SPLIT_FILE, 'w') as f_out:
        for base in imagenes_validas:
            f_out.write(test_basenames[base] + "\n")

    print("-" * 30)
    print(f"[*] ¡PROCESO UNIFICADO COMPLETADO!")
    print(f"[*] Imágenes procesadas y listas para inferencia: {len(imagenes_validas)}")
    print(f"[*] Guardadas en: {OUTPUT_IDS_DIR}")
    print(f"[*] Split creado: {NEW_SPLIT_FILE}")

if __name__ == "__main__":
    main()