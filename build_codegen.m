% Configuración: librería estática C++ para Debian aarch64 (QEMU)
cfg = coder.config('lib');
cfg.TargetLang     = 'C++';
cfg.GenCodeOnly    = true;   % solo fuentes; la compilación se hace en Debian

% Backend de deep learning portable (MKL-DNN es solo x86 Intel, no sirve en ARM)
cfg.DeepLearningConfig = coder.DeepLearningConfig('TargetLibrary','none');

% Target: ARM Cortex-A / Linux (aarch64)
cfg.HardwareImplementation.ProdHWDeviceType = 'ARM Compatible->ARM Cortex-A';
cfg.HardwareImplementation.ProdLongLongMode = true;
cfg.HardwareImplementation.ProdEndianess    = 'LittleEndian';

% Tipos de entrada — AJUSTÁ [H W 3] al tamaño real de tu imagen
imgType = coder.typeof(uint8(0), [720 1280 3]);         % fijo
% imgType = coder.typeof(uint8(0), [1080 1920 3], [1 1 0]);  % variable

matType = coder.Constant('yoloDetector.mat');

% Generar
codegen -config cfg yolov2_detect -args {imgType, matType} -report
