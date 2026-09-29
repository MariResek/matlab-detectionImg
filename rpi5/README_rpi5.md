# YOLOv2 en Raspberry Pi 5

Corre el mismo detector que armamos para QEMU, pero **de forma nativa** en
un Raspberry Pi 5 (aarch64, Cortex-A76). Inferencia de segundos → menos de
un segundo por imagen.

## Requisitos

- Raspberry Pi 5 (4 GB u 8 GB de RAM).
- **Raspberry Pi OS 64-bit** (Debian 12 Bookworm 64-bit) o Debian aarch64.
- Acceso por SSH desde tu PC (activarlo con `sudo raspi-config` →
  Interface Options → SSH).
- Los archivos `.mat` y el codegen ya generados en Windows con MATLAB
  Coder (ver el README principal del proyecto).

## Paso a paso

### 1) Preparar la carpeta en el host (Windows)

Copiá `cpp_debian/` a un nombre nuevo llamado `yolov2` y agregale estos
tres scripts:

```
yolov2\
    main.cpp
    Makefile
    yolov2_detect\      (fuentes + .bin generados por MATLAB Coder)
    setup_rpi5.sh
    build_rpi5.sh
    run_rpi5.sh
```

No hace falta bajar `stb_image.h` a mano: `setup_rpi5.sh` los baja solo
con `curl` en la Pi.

### 2) Enviarlo a la Raspberry Pi

Averiguá la IP del Pi (`hostname -I` en el Pi, o mirá tu router). Desde
PowerShell del host:

```bash
scp -r yolov2 pi@raspberrypi.local:~/
```

(Reemplazá `pi@raspberrypi.local` por tu usuario e IP, ej `pi@192.168.1.42`).

### 3) En la Raspberry Pi: setup, build, run

Entrá por SSH:

```bash
ssh pi@raspberrypi.local
cd ~/yolov2
chmod +x setup_rpi5.sh build_rpi5.sh run_rpi5.sh

./setup_rpi5.sh     # una sola vez (instala g++, imagemagick, baja stb, etc.)
./build_rpi5.sh     # una sola vez (o cada vez que cambies main.cpp)
./run_rpi5.sh gato.png resultado.png
```

Salida esperada:

```
==> Imagen 1900x1400 != 1280x720. Redimensionando...
==> Corriendo detect...
Inicializando red...
Corriendo yolov2_detect...
Detecciones: 1
  #0  lb=16  cat         score=0.980  bbox=[296.0 131.0 691.0 559.0]
Guardado: resultado.png

real    0m1.234s   <-- ¡segundos, no minutos!
```

### 4) Traer el resultado

Desde tu PC:

```bash
scp pi@raspberrypi.local:~/yolov2/resultado.png C:\Users\<TU>\Desktop\
```

## Diferencias con la versión de QEMU

| Aspecto              | QEMU aarch64             | Raspberry Pi 5                     |
| -------------------- | ------------------------ | ---------------------------------- |
| Velocidad            | Minutos por imagen       | < 1 segundo (fresh) / ~500 ms      |
| Threads (`OMP`)      | Forzado a 1              | 4 (`$(nproc)`), habilitados por `run_rpi5.sh` |
| Flags de compilación | Genéricos                | `-mcpu=cortex-a76` (SIMD NEON del A76) |
| Stack overflow       | Sí (`ulimit unlimited`)  | Igual — el setup lo pone en `.bashrc` |

## Aún más rápido: ARM Compute Library

Si querés máxima performance, regenerá el codegen en MATLAB con:

```matlab
cfg.DeepLearningConfig = coder.DeepLearningConfig('arm-compute');
```

Y en el Pi instalá ARM Compute Library:

```bash
sudo apt install -y libarm-compute24  # nombre puede variar
```

Con eso las convoluciones usan implementaciones NEON altamente
optimizadas (2–5× más rápido). El main.cpp NO cambia; solo cambia lo
que compila `make`.

## Cámara del RPi (opcional, próximo paso)

Para capturar de la Pi Camera y detectar en tiempo real, usá `libcamera`:

```bash
libcamera-still -o frame.jpg --width 1280 --height 720
./run_rpi5.sh frame.jpg out.png
```

Podemos armar un loop en shell o portar `main.cpp` para leer directo del
sensor con la API `libcamera`; decilo y lo agregamos.

## Errores comunes

- **`Illegal instruction`** al correr `./detect`: el binario se compiló
  con flags de un CPU distinto. Reeditá `build_rpi5.sh` cambiando
  `-mcpu=cortex-a76` por `-mcpu=native` o simplemente `-march=armv8-a`,
  y reebuild.
- **`No such file or directory: ./codegen/lib/yolov2_detect/…bin`**: no
  corriste `setup_rpi5.sh` o falta `yolov2_detect/*.bin`. Reejecutalo.
- **`Killed`** al mitad de la inferencia: te quedaste sin RAM. Cerrá
  aplicaciones y volvé a probar; o creá swap:
  ```bash
  sudo dphys-swapfile swapoff
  sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
  sudo dphys-swapfile setup
  sudo dphys-swapfile swapon
  ```
