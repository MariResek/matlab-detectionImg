Fase 2 — Preparar el código para codegen

El código que se convierte a C++ no puede ser tu script tal cual; tiene que ser una función de entrada (entry-point) que cumpla reglas de codegen.

Escribí una función tipo detectObjects(img) que cargue la red con coder.loadDeepLearningNetwork y llame a detect.
Nada de imshow, VideoReader, ni gráficos adentro de la función que generás (esos van en el wrapper de C++, o leés frames de otra forma).
Investigá: doc "Code Generation for Object Detection by Using YOLO v2" — es el ejemplo oficial que hace exactamente tu pipeline. Y "Entry-Point Functions and Code Generation".







MATLAB Coder no genera código C/C++ directamente desde un script. Necesita una función que sea el punto de entrada del código generado.
La función detectObjects sera nuestra interfaz entre tu futuro programa C++ y MATLAB generado.

Para ello creamos la funcion: detectObjects.m


En la documentacion esta asi:

function outImg = yolov2_detect(in,matFile)

persistent yolov2Obj;

if isempty(yolov2Obj)
    yolov2Obj = coder.loadDeepLearningNetwork(matFile);
end

% Call to detect method
[bboxes,~,labels] = yolov2Obj.detect(in,Threshold=0.5);

% Convert categorical labels to cell array of charactor vectors
labels = cellstr(labels);

% Annotate detections in the image.
outImg = insertObjectAnnotation(in,"rectangle",bboxes,labels);

este es yolov2_detect.m
%lo que yo voy a resumir:

function [bboxes, scores, labels] = yolov2_detect(in, netMatFile)
%#codegen

persistent yolov2Obj;

if isempty(yolov2Obj)
    yolov2Obj = coder.loadDeepLearningNetwork(netMatFile);
end

[bboxes, scores, labelsCat] = detect(yolov2Obj, in, Threshold=0.4);

% Convertir categorical a índices uint32 (compatible con C++)
labels = uint32(labelsCat);

end

%RedmatFile debe contener el yolo-v2 
% y se lo va a llamar asi 
in = imread("foto.jpg");
outImg = yolov2_detect(in, "miRed.mat");

Crear el archivo prueba.m

I = imread('la-direccion-de-tu-imagen');

[bboxes, scores, labels] = yolov2_detect(I, 'yolov2Detector.mat');

Iout = insertObjectAnnotation(I, ...
    "rectangle", ...
    bboxes, ...
    labels);

imshow(Iout);



Para lo proximo debemos tener:
Computer Vision Toolbox
Deep Learning Toolbox
MATLAB Coder
GPU Coder

en terminal para crear el yoloDetector.mat
save('yoloDetector.mat', 'detector'); %matlab ya trae uno preentrenado

antes de ejecutar prueba.m 
en terminal de matlab
detector = yolov2ObjectDetector("tiny-yolov2-coco");
save('yoloDetector.mat', 'detector');
clear detector
para ver si realmente se vuelve a cargar desde un comienzo

net = coder.loadDeepLearningNetwork('yoloDetector.mat');
I = imread('C:\Users\mrese\OneDrive\Escritorio\INFORMATICA\MATLAB\gato.png');

[bboxes, scores, labels] = detect(net, I, Threshold=0.4);
labels

[bboxes, scores, labels] = yolov2_detect( ...
    I, 'yoloDetector.mat');

Y ya tenemos        yoloDetector.mat
                           ↓
              coder.loadDeepLearningNetwork
                           ↓
                       YOLOv2
                           ↓
                        detect
                           ↓
              ┌────────────┼────────────┐
              ↓            ↓            ↓
           bboxes       scores       labels

Aclaracion:
tirar en terminal matlab
detector = yolov2ObjectDetector("tiny-yolov2-coco");
save('yoloDetector.mat', 'detector');
clear detector
net = coder.loadDeepLearningNetwork('yoloDetector.mat');
correr probar.m


corroborar la imagen 
whos I
Preparar argumento de entrada
input = zeros(size(I), 'uint8');
Ahora input tiene exactamente el mismo tamaño y tipo que tu imagen.

codegen -config:lib yolov2_detect -args {input, coder.Constant('yoloDetector.mat')}

compila con 
codegen -config:lib yolov2_detect -args {input, coder.Constant('yoloDetector.mat')}

instalar 
matlab.addons.supportpackage.internal.explorer.showSupportPackages('ML_DEEPLEARNING_LIB', 'tripwire')
y ver con:
matlab.addons.installedAddons


test antes de compilar 
I = imread('C:\Users\mrese\OneDrive\Escritorio\INFORMATICA\MATLAB\gato.png');
[bboxes, scores, labelsIdx] = yolov2_detect(I, 'yoloDetector.mat');


crear el archivo generar_cpp.m

% Configuración: librería estática C++
cfg = coder.config('lib');
cfg.TargetLang = 'C++';
cfg.GenCodeOnly = false;

% Backend de deep learning (elegí uno):
cfg.DeepLearningConfig = coder.DeepLearningConfig('TargetLibrary','none');
% cfg.DeepLearningConfig = coder.DeepLearningConfig('mkldnn');  % CPU Intel más rápido

% Tipos de entrada — AJUSTÁ [H W 3] al tamaño real de tu imagen
imgType = coder.typeof(uint8(0), [720 1280 3]);         % fijo
% imgType = coder.typeof(uint8(0), [1080 1920 3], [1 1 0]);  % variable hasta ese máx

matType = coder.Constant('yoloDetector.mat');

% Generar
codegen -config cfg yolov2_detect -args {imgType, matType} -report





% Recuperar nombres para visualizar
tmp = load('yoloDetector.mat');
detector = tmp.(char(fieldnames(tmp)));   % o el nombre real de la variable
classNames = detector.ClassNames;
labelsStr = cellstr(classNames(labelsIdx));

Iout = insertObjectAnnotation(I, "rectangle", bboxes, labelsStr, 'FontSize', 50);
imshow(Iout);