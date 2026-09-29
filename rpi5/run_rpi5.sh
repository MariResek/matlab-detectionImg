#!/usr/bin/env bash
# run_rpi5.sh
# Wrapper para correr el detector.
#
# Uso:
#   ./run_rpi5.sh <imagen_entrada> [imagen_salida]
#
# Si la imagen no es 1280x720, la redimensiona automáticamente a
# ~/yolov2/_tmp_input.png antes de correr.

set -euo pipefail

cd "$(dirname "$0")"

if [[ $# -lt 1 ]]; then
    echo "Uso: $0 <imagen_entrada> [imagen_salida]"
    exit 1
fi

IN_IMG="$1"
OUT_IMG="${2:-out.png}"

if [[ ! -f "$IN_IMG" ]]; then
    echo "!! No existe: $IN_IMG"
    exit 1
fi

# Verificar tamaño con ImageMagick 'identify' y redimensionar si hace falta
DIMS=$(identify -format "%w %h" "$IN_IMG")
W=$(echo "$DIMS" | awk '{print $1}')
H=$(echo "$DIMS" | awk '{print $2}')

if [[ "$W" -ne 1280 || "$H" -ne 720 ]]; then
    echo "==> Imagen ${W}x${H} != 1280x720. Redimensionando..."
    TMP="_tmp_input.png"
    convert "$IN_IMG" -resize 1280x720! "$TMP"
    IN_IMG="$TMP"
fi

# OpenMP: usar los 4 cores del RPi5. Si notás inestabilidad, bajá a 1.
export OMP_NUM_THREADS="$(nproc)"
export OMP_STACKSIZE=512M
ulimit -s unlimited

echo "==> Corriendo detect..."
time ./detect "$IN_IMG" "$OUT_IMG"

echo ""
echo "==> Resultado guardado en: $OUT_IMG"
