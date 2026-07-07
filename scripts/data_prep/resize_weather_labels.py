#!/usr/bin/env python3
"""
Resize de las mascaras del dataset de clima para que CASEN con las imagenes.
- Imagenes: 1280x720 (NO se tocan).
- Mascaras (labels, labels_color, instances): 1920x1080 -> 1280x720 con NEAREST
  (no mezcla ids de clase ni colores; mismo aspect ratio 16:9, sin distorsion).
- No destructivo: escribe en <carpeta>_720/ y conserva los originales.
- Comprueba que labels/ tiene solo valores de clase 0..8 (+255 ignore).

Correr DENTRO del contenedor road_defect_base (tiene PIL/numpy):
  docker run --rm -v /home/lpa/Documentos/tfm:/app -w /app \
    road_defect_base:latest python3 scripts/data_prep/resize_weather_labels.py
"""
import glob, os
import numpy as np
from PIL import Image

DATA = "/app/data/final_dataset"
TARGET = (1280, 720)                       # (W, H) = tamano de las imagenes
FOLDERS = ["labels", "labels_color", "instances"]
NUM_CLASSES = 9

for folder in FOLDERS:
    src = f"{DATA}/{folder}"
    dst = f"{DATA}/{folder}_720"
    files = sorted(glob.glob(f"{src}/*.png"))
    if not files:
        print(f"[skip] {folder}: sin PNGs en {src}")
        continue
    os.makedirs(dst, exist_ok=True)
    sizes_in, class_vals = set(), set()
    for p in files:
        im = Image.open(p)
        sizes_in.add(im.size)              # (W, H)
        out = im.resize(TARGET, Image.Resampling.NEAREST)
        out.save(f"{dst}/{os.path.basename(p)}")
        if folder == "labels":
            class_vals |= set(np.unique(np.asarray(out)).tolist())
    print(f"[ok] {folder}: {len(files)} PNG  {sorted(sizes_in)} -> {TARGET}  =>  {folder}_720/")
    if folder == "labels":
        bad = sorted(v for v in class_vals if v > NUM_CLASSES - 1 and v != 255)
        estado = "OK (subset 0..8 [+255])" if not bad else f"*** OJO valores raros: {bad} ***"
        print(f"      valores de clase en labels: {sorted(class_vals)}  ->  {estado}")

print("DONE")
