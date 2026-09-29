# Detector YOLOv2 en MATLAB → C++ para Debian ARM64 (QEMU)

Este proyecto entrena/usa un detector de objetos YOLOv2 en MATLAB y lo
convierte a código C++ portable para correrlo en una VM Debian aarch64
emulada con QEMU. La detección se hace 100 % en C++ (sin MATLAB en la VM).

> **En qué consiste**: partís de una red YOLOv2 (COCO, 80 clases) entrenada
> en MATLAB, generás automáticamente el código C++ equivalente con MATLAB
> Coder, lo copiás a una VM Debian ARM64, lo compilás con `g++` y corrés
> un ejecutable que toma una imagen y devuelve otra con los objetos
> detectados marcados con rectángulos.

---

## Archivos del proyecto

| Archivo/carpeta                | Qué es                                                                                       |
| ------------------------------ | -------------------------------------------------------------------------------------------- |
| `yoloDetector.mat`             | Red YOLOv2 pre-entrenada (COCO, 80 clases). Cargá con `load('yoloDetector.mat')` en MATLAB.  |
| `yolov2_detect.m`              | Función `#codegen` que envuelve la inferencia. Es lo que MATLAB Coder traduce a C++.         |
| `probar.m`                     | Script MATLAB para probar la detección sobre `gato.png` (referencia; **no** se usa en C++).  |
| `build_codegen.m`              | Script que llama a MATLAB Coder para generar el código C++.                                  |
| `gato.png`                     | Imagen de prueba (un gato).                                                                  |
| `cpp_debian/main.cpp`          | Programa C++ que invoca la red generada, dibuja los bboxes y guarda el resultado.            |
| `cpp_debian/Makefile`          | Reglas para compilar todo en Debian aarch64 con `g++`.                                       |
| `codegen/lib/yolov2_detect/`   | **Se crea automáticamente** al correr `build_codegen`. Contiene los `.cpp/.h` de la red.     |

Falta descargar dos headers (paso 3.b más abajo):
- `cpp_debian/stb_image.h`
- `cpp_debian/stb_image_write.h`

---

## Requisitos

### En Windows (host)
- **MATLAB R2022b o superior** con:
  - MATLAB Coder
  - Deep Learning Toolbox
  - Computer Vision Toolbox
  - MATLAB Support Package for Deep Learning Networks (para YOLOv2)
- **QEMU** para Windows (`qemu-system-aarch64.exe`).
- **OpenSSH client** (viene por defecto en Windows 10/11).
- Cliente **PowerShell**.

### VM Debian ARM64 (aarch64) en QEMU
- Debian 12 (trixie) o superior, aarch64.
- **Recomendado**: 4 GB RAM, 2 vCPU, 16 GB de disco.
- Redirección de puerto **2222 → 22** para SSH desde el host.
- Usuario con `sudo`. En este proyecto usamos `mresek`; adaptá si es otro.

Ejemplo de arranque de QEMU con SSH redirigido:

```bat
qemu-system-aarch64.exe ^
  -M virt -cpu cortex-a72 -smp 2 -m 4G ^
  -bios QEMU_EFI.fd ^
  -drive if=none,file=debian.qcow2,id=hd0 ^
  -device virtio-blk-pci,drive=hd0 ^
  -netdev user,id=n1,hostfwd=tcp::2222-:22 ^
  -device virtio-net-pci,netdev=n1 ^
  -nographic
```

---

## Paso a paso completo

### 1) Generar el código C++ desde MATLAB (Windows)

Abrí MATLAB y posicionate en la carpeta del proyecto:

```matlab
cd 'C:\Users\<TU_USUARIO>\...\MATLAB'
```

Ajustá `build_codegen.m` si tu imagen NO es 1280×720. Cambiá la línea:

```matlab
imgType = coder.typeof(uint8(0), [720 1280 3]);   % [H W 3]
```

Corré:

```matlab
build_codegen
```

Puede tardar **5 a 15 minutos** la primera vez. No cortes con Ctrl+C
aunque parezca colgado. Al terminar aparece:

```
Code generation successful: View report
```

Se crea la carpeta `codegen\lib\yolov2_detect\` con montones de
`.cpp`, `.h` y archivos `.bin` (pesos de la red).

### 2) Preparar el compilador dentro de Debian (una sola vez)

Desde la consola de la VM (o por SSH):

```bash
sudo apt update
sudo apt install -y g++ make openssh-server strace imagemagick
sudo systemctl enable --now ssh
```

- `g++` y `make` para compilar.
- `openssh-server` para poder entrar por SSH y usar `scp`.
- `strace` para debug si falla.
- `imagemagick` da el comando `convert` para redimensionar imágenes.

### 3) Preparar la carpeta `cpp_debian` en Windows

#### 3.a) Copiar los fuentes generados

En el Explorador de Windows:
1. Ir a `codegen\lib\yolov2_detect\`
2. Copiar todo el contenido (Ctrl+A → Ctrl+C).
3. Ir a `cpp_debian\`, crear una carpeta llamada `yolov2_detect`,
   entrar, pegar (Ctrl+V).

Al final `cpp_debian\yolov2_detect\` debe tener `yolov2_detect.h`,
`yolov2_detect.cpp`, y decenas de archivos más, incluyendo varios `.bin`.

#### 3.b) Descargar los headers de `stb`

Son dos archivos "single header" para leer/escribir imágenes sin OpenCV:

- https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
- https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h

Guardá cada uno directamente en `cpp_debian\` como:
- `stb_image.h`
- `stb_image_write.h`

Cuidado: Windows a veces les agrega `.txt` al final. Verificalo con
Ver → Mostrar → Extensiones de nombre de archivo, y si están como
`stb_image.h.txt`, renombralos quitando el `.txt`.

Al final `cpp_debian\` debe tener:

```
cpp_debian\
    main.cpp
    Makefile
    stb_image.h
    stb_image_write.h
    yolov2_detect\          (con muchos .cpp/.h/.bin adentro)
```

### 4) Copiar `cpp_debian` a la VM

En **PowerShell** del host (ajustá el usuario y la ruta):

```bash
cd C:\Users\<TU_USUARIO>\...\MATLAB
scp -P 2222 -r cpp_debian mresek@localhost:/home/mresek/
```

Te pide la contraseña de `mresek` en Debian.

### 5) Preparar los pesos de la red en Debian

MATLAB Coder hardcodea la ruta relativa `./codegen/lib/yolov2_detect/`
para leer los `.bin`. Hay que replicarla:

```bash
ssh -p 2222 mresek@localhost
cd ~/cpp_debian
mkdir -p codegen/lib/yolov2_detect
cp yolov2_detect/*.bin codegen/lib/yolov2_detect/
ls codegen/lib/yolov2_detect/
```

Debe listar 8 archivos `.bin`.

### 6) Compilar

Adentro de Debian, en la misma sesión SSH:

```bash
cd ~/cpp_debian
make -j2
```

Va a tardar bastante (muchos archivos + QEMU emula CPU). Al terminar
hay un ejecutable llamado `detect` en la carpeta.

### 7) Preparar la imagen de entrada

`main.cpp` espera exactamente **1280×720** (el tamaño con el que
`build_codegen` generó la red). Si tu imagen es otra medida,
redimensionala dentro de Debian:

```bash
convert gato.png -resize 1280x720! gato_720p.png
```

El `!` fuerza el tamaño exacto (deforma el aspecto pero YOLO lo tolera).

### 8) Correr la detección

```bash
cd ~/cpp_debian
ulimit -s unlimited
export OMP_NUM_THREADS=1
export OMP_STACKSIZE=512M
./detect gato_720p.png out.png
```

Salida esperada:

```
Inicializando red...
Corriendo yolov2_detect... (puede tardar mucho en QEMU)
Detecciones: 1
  #0  lb=16  cat         score=0.980  bbox=[296.0 131.0 691.0 559.0]
Guardado: out.png
```

La primera vez tarda **varios minutos** (QEMU emula la CPU ARM sobre
x86). Corridas siguientes son igualmente lentas — no hay atajo.

Para no repetir los `ulimit` y `export` cada vez, añadilos a `.bashrc`:

```bash
echo 'ulimit -s unlimited' >> ~/.bashrc
echo 'export OMP_NUM_THREADS=1' >> ~/.bashrc
echo 'export OMP_STACKSIZE=512M' >> ~/.bashrc
```

### 9) Traer el resultado al host y verlo

En **PowerShell** del host:

```bash
scp -P 2222 mresek@localhost:/home/mresek/cpp_debian/out.png C:\Users\<TU_USUARIO>\Desktop\
```

Doble click y verás la imagen con rectángulos verdes alrededor de los
objetos detectados.

---

## Cómo probar con otras imágenes

```bash
# Copiar la imagen al Debian
scp -P 2222 C:\ruta\a\otra.jpg mresek@localhost:/home/mresek/cpp_debian/

# En Debian
cd ~/cpp_debian
convert otra.jpg -resize 1280x720! otra_720p.png
./detect otra_720p.png otra_out.png
scp -P 2222 mresek@localhost:/home/mresek/cpp_debian/otra_out.png C:\Users\<TU>\Desktop\
```

---

## Cómo cambiar el tamaño de red aceptado

Si tus imágenes son siempre otro tamaño (por ejemplo 640×480),
editá `build_codegen.m`:

```matlab
imgType = coder.typeof(uint8(0), [480 640 3]);
```

Y editá `cpp_debian/main.cpp`, las líneas:

```cpp
static constexpr int H = 720;
static constexpr int W = 1280;
```

por los nuevos valores. Después:

1. Volvé a correr `build_codegen` en MATLAB.
2. Repetí los pasos 3.a (copiar la nueva carpeta `codegen\lib\yolov2_detect`)
   y 4 (scp a Debian).
3. `make clean && make -j2` en Debian.

Alternativa: usar tamaño variable con
`coder.typeof(uint8(0), [H_max W_max 3], [1 1 0])` y adaptar `main.cpp`
para leer las dimensiones de `stbi_load`. Es más flexible pero requiere
regenerar cada vez que cambies el máximo.

---

## Errores comunes y cómo resolverlos

### "Connection refused" al hacer `ssh -p 2222`
- QEMU no está corriendo.
- QEMU corre pero sin `-hostfwd=tcp::2222-:22`.
- SSH no está encendido en Debian: `sudo systemctl enable --now ssh`.

### `fatal error: stb_image.h: No such file or directory`
- Los headers están como `stb_image.h.txt`. Renombralos.

### `undefined reference to 'omp_init_nest_lock'`
- Falta `-fopenmp` en el Makefile (ya está incluido; si lo borraste, restauralo).

### `Tamano de imagen (WxH) != esperado (1280x720)`
- Redimensioná la imagen con `convert IMG -resize 1280x720! OUT.png`.

### `./detect` sale con `exit code: 1` inmediatamente
- Falta la carpeta `codegen/lib/yolov2_detect/` con los `.bin` al lado del binario.
  Reejecutá el paso 5.
- Si no, corré con `gdb` para ver dónde muere:
  ```bash
  gdb -batch -ex "break exit" -ex run -ex "bt 40" --args ./detect gato_720p.png out.png
  ```

### Todas las labels salen como `?`
- El orden de `CLASS_NAMES[]` en `main.cpp` no coincide con
  `detector.ClassNames` del `.mat`. Confirmá en MATLAB con:
  ```matlab
  load('yoloDetector.mat'); disp(detector.ClassNames)
  ```
  y ajustá el array en `main.cpp`.

### El detector no encuentra nada aunque el objeto está claramente en la imagen
- Bajá el umbral: en `yolov2_detect.m`, cambiar `Threshold=0.4` por `Threshold=0.2`.
- Regenerá el codegen.

---

## Rendimiento

- QEMU aarch64 emula cada instrucción ARM en x86 → **muy lento**.
- Con `TargetLibrary='none'` (backend portable), esperar **varios
  minutos por imagen**. Es correcto.
- Para acelerar en un ARM real (Raspberry Pi, Jetson, etc.), regenerar
  con:
  ```matlab
  cfg.DeepLearningConfig = coder.DeepLearningConfig('arm-compute');
  ```
  Requiere tener instalada ARM Compute Library en el target.

---

## Licencia

Los archivos generados por MATLAB Coder incluyen una nota "Academic
License" — uso académico y de investigación únicamente. Si querés
usarlo comercialmente, necesitás la licencia correspondiente de
MathWorks.
