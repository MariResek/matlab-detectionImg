#!/usr/bin/env bash
# build_rpi5.sh
# Compila el detector en Raspberry Pi 5. Usa los 4 cores del Cortex-A76
# y aprovecha NEON.

set -euo pipefail

cd "$(dirname "$0")"

# -mcpu=cortex-a76: instrucciones específicas del RPi5 (mejor rendimiento
# que el genérico armv8-a). Si tenés problemas de compatibilidad,
# reemplazar por -mcpu=native o borrar.
export CXXFLAGS="-O3 -std=c++14 -fPIC -fopenmp -mcpu=cortex-a76 \
                 -Wno-unused-variable -Wno-unused-but-set-variable"

make clean
make -j"$(nproc)"

echo ""
echo "==> Listo. Ejecutar con:"
echo "     ./run_rpi5.sh <imagen>"
