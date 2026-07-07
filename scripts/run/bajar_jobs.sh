#!/usr/bin/env bash
# ===========================================================================
# Descarga SECUENCIAL de jobs de Azure ML (logs + outputs + checkpoints).
# Cada job se descarga en data/checkpoints/<modelo>/seed_<N>/.
# - Continúa aunque uno falle (no aborta la cola).
# - Es RE-EJECUTABLE: salta los que ya tienen un .pth descargado (resume).
# - Salta las celdas cuyo nombre de job esté vacío (placeholder sin rellenar).
#
# Uso normal:
#     bash scripts/run/bajar_jobs.sh
# Desatendido (sobrevive cerrar la terminal):
#     nohup bash scripts/run/bajar_jobs.sh > data/checkpoints/_download.log 2>&1 &
#     tail -f data/checkpoints/_download.log     # Ctrl-C solo corta el tail
# ===========================================================================
set -uo pipefail
# run from repo root regardless of CWD (this script lives in scripts/run/)
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1

# Azure coordinates from the gitignored .env (see .env.example)
[ -f .env ] && { set -a; . ./.env; set +a; }
RG="${AZ_RESOURCE_GROUP:?set AZ_RESOURCE_GROUP in .env}"
WS="${AZ_WORKSPACE:?set AZ_WORKSPACE in .env}"
OUT=data/checkpoints                 # carpeta raíz de descargas

# ---------------------------------------------------------------------------
# RELLENA el nombre de job de cada celda (job hijo de 'segmentation', o el
# padre si baja los outputs).  Formato:  "NOMBRE_JOB|modelo/seed_XX"
# Deja el nombre vacío ("|modelo/seed_XX") o comenta (#) los que no tengas aún.
# ---------------------------------------------------------------------------
jobs=(
  # --- BEiT2 ---
  # "zen_scooter_ybmc5rkdk9|beit/seed_91"        # (ya descargado)
  #"willing_station_tt4hlbrkzd|beit/seed_1337"
  "funny_rain_0q7kld0zhm|beit/seed_42"
  #"|beit/seed_777"
  #"|beit/seed_2026"

  # --- Swin ---
  "bubbly_cart_qpqn6mygc3|swin/seed_42"
  "tough_drain_671qflbgcl|swin/seed_91"
  "happy_stone_sm2mnh8n27|swin/seed_777"
  "upbeat_apricot_qvxr7n26wv|swin/seed_1337"
  "placid_knot_gth13qtwl5|swin/seed_2026"

  # --- HRNet ---
  "bubbly_dog_0wl4zcs99k|hrnet/seed_42"
  "maroon_rail_2q6ym25kkl|hrnet/seed_91"
  "modest_arch_zkg8y9m0th|hrnet/seed_777"
  "willing_station_tt4hlbrkzd|hrnet/seed_1337"
  "crimson_cherry_gb6p059fv2|hrnet/seed_2026"

  # --- InterImage ---
  "bold_rocket_vp9t1x82tm|interimage/seed_42"
  "lime_thread_9x1yrry0td|interimage/seed_91"
  "joyful_garage_ccwb2fxzkq|interimage/seed_777"
  "magenta_lime_19hglz0v6t|interimage/seed_1337"
  "mango_van_zwb38n5kk8|interimage/seed_2026"

  # --- FlashInternImage ---
  "silly_lunch_9j1stpjz93|flash/seed_42"
  "sad_angle_d0q95n3f29|flash/seed_91"
  "red_rail_rthhv4y8v8|flash/seed_777"
  "amusing_flower_fq9ffpdth6|flash/seed_1337"
  "joyful_nail_xp2rv6ql16|flash/seed_2026"
)

ts() { date +%H:%M:%S; }
mkdir -p "$OUT"
total=0; ok=0; fail=0; skipped=0

echo "=== [$(ts)] Inicio de la cola de descargas -> $OUT ==="
for e in "${jobs[@]}"; do
  name="${e%%|*}"
  dest="$OUT/${e##*|}"

  # placeholder sin rellenar -> saltar en silencio
  [ -z "$name" ] && continue
  total=$((total+1))

  # resume: si ya hay un .pth en el destino, no re-descargar
  if [ -n "$(find "$dest" -name '*.pth' 2>/dev/null | head -1)" ]; then
    echo "=== [$(ts)] SKIP  $name  (ya hay .pth en $dest) ==="
    skipped=$((skipped+1)); continue
  fi

  echo "=== [$(ts)] START $name -> $dest ==="
  mkdir -p "$dest"
  if az ml job download -n "$name" --all -p "$dest" -g "$RG" -w "$WS"; then
    echo "=== [$(ts)] OK    $name ==="
    ok=$((ok+1))
  else
    echo "=== [$(ts)] FALLO $name  (sigo con el siguiente) ==="
    fail=$((fail+1))
  fi
done

echo
echo "=== [$(ts)] TERMINADO ==="
echo "    jobs con nombre: $total   OK: $ok   fallidos: $fail   saltados(ya estaban): $skipped"
echo
echo "=== checkpoints .pth descargados ==="
find "$OUT" -name '*.pth' 2>/dev/null | sort
