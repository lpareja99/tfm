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
# Fill in one entry per (model, seed) with your Azure ML job name.
#   Format:  "JOB_NAME|model/seed_XX"   (leave empty "|model/seed_XX" or comment
#   out the ones you don't have yet). The author's filled list (job names are
#   workspace-internal) is kept in the gitignored README.local.md — paste it here.
# ---------------------------------------------------------------------------
jobs=(
  # --- BEiT2 ---
  # "<job-name>|beit/seed_42"
  # --- Swin ---
  # "<job-name>|swin/seed_91"
  # --- HRNet ---
  # "<job-name>|hrnet/seed_1337"
  # --- InterImage ---
  # "<job-name>|interimage/seed_91"
  # --- FlashInternImage ---
  # "<job-name>|flash/seed_777"
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
