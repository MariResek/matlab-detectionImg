#!/usr/bin/env bash
# setup_rpi5.sh
# Instala dependencias y prepara el entorno en Raspberry Pi 5 (Raspberry Pi OS
# 64-bit o Debian aarch64). Correr una sola vez con:
#   chmod +x setup_rpi5.sh && ./setup_rpi5.sh
#
# Estructura esperada tras "deploy" desde el host:
#   ~/yolov2/
#       main.cpp
#       Makefile
#       stb_image.h            (se baja automáticamente aquí si falta)
#       stb_image_write.h      (idem)
#       yolov2_detect/         (fuentes generados por MATLAB Coder + .bin)

set -euo pipefail

PROJECT_DIR="${HOME}/yolov2"

echo "==> Actualizando índice de paquetes..."
sudo apt update

echo "==> Instalando compilador y utilidades..."
sudo apt install -y \
    g++ make \
    imagemagick \
    curl \
    strace gdb

if [[ ! -d "$PROJECT_DIR" ]]; then
    echo "!! No existe $PROJECT_DIR."
    echo "   Copiá primero la carpeta cpp_debian (renombrada 'yolov2') desde el host."
    echo "   Ejemplo desde Windows con scp:"
    echo "     scp -r cpp_debian pi@raspberrypi.local:~/yolov2"
    exit 1
fi

cd "$PROJECT_DIR"

echo "==> Bajando headers stb si faltan..."
if [[ ! -f stb_image.h ]]; then
    curl -sSL -o stb_image.h \
        https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
fi
if [[ ! -f stb_image_write.h ]]; then
    curl -sSL -o stb_image_write.h \
        https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h
fi

echo "==> Preparando pesos de la red en la ruta que espera el código..."
if [[ ! -d yolov2_detect ]]; then
    echo "!! Falta la carpeta yolov2_detect/. Copiala del host antes de continuar."
    exit 1
fi
mkdir -p codegen/lib/yolov2_detect
cp -u yolov2_detect/*.bin codegen/lib/yolov2_detect/ 2>/dev/null || true
ls codegen/lib/yolov2_detect/*.bin | head

echo "==> Ajustando límites del shell de forma permanente..."
if ! grep -q 'ulimit -s unlimited' "${HOME}/.bashrc"; then
    {
        echo ''
        echo '# YOLOv2 codegen necesita stack grande + OpenMP con stack grande'
        echo 'ulimit -s unlimited'
        echo 'export OMP_STACKSIZE=512M'
    } >> "${HOME}/.bashrc"
fi

echo "==> Todo listo. Ahora corré:"
echo "     cd ~/yolov2 && ./build_rpi5.sh"
