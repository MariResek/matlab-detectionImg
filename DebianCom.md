
qemu-system-aarch64 -machine virt -cpu cortex-a72 -smp 2 -m 4096 -drive if=pflash,format=raw,readonly=on,file="C:\Program Files\qemu\share\edk2-aarch64-code.fd" -drive format=qcow2,file="C:\QEMU\debian\debian-arm64.qcow2" -nographic -netdev user,id=net0,hostfwd=tcp::2222-:22 -device virtio-net-pci,netdev=net0

Conectarme qemu por ssh

ssh mresek@localhost -p 2222

apagar la VM 
sudo shutdown -h now

Si la terminal la queda:

Probá esto ahora, en esa misma ventana:

Apretá Enter un par de veces — a veces solo falta redibujar el prompt.
Si no aparece nada, escribí a ciegas (aunque no veas nada):
   stty sane

y Enter. Eso resetea la terminal.


Copiar desde Windows a la VM

powershell
scp -P 2222 "C:\ruta\a\tu\imagen.png" mresek@localhost:/home/mresek/

Te pide la contraseña de mresek y copia el archivo a /home/mresek/.

Ojo: en scp es -P mayúscula (en ssh es -p minúscula, es un detalle molesto).

Copiar desde la VM a Windows

powershell
scp -P 2222 mresek@localhost:/home/mresek/archivo.png "C:\ruta\destino\"

y desde debian reviar
sudo apt update
sudo apt install -y openssh-server
sudo systemctl enable --now ssh
systemctl status ssh



para compilar en debian
sudo apt update
sudo apt install -y g++ make

cd ~/cpp_debian
ls
y por ultimo 
make


correrlo 
cd ~/cpp_debian
./detect gato.png out.png


el tema que tuve que redimensionar
cd ~/cpp_debian
./detect gato_720p.png out.png


conectarme por ssh:
ssh -p 2222 mresek@localhost
