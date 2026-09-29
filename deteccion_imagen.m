% deteccion_imagen.m
% Fase 1 - Detección de objetos en una imagen fija con tiny-YOLOv2 (COCO).

clear; clc;

% 1) Cargar el detector preentrenado (la 1a vez MATLAB baja el support package)
detector = yolov2ObjectDetector('tiny-yolov2-coco');

% 2) Cargar una imagen de prueba que viene con MATLAB
%   imread('C:\ruta\a\tu_foto.jpg')
I = imread('C:\Users\mrese\OneDrive\Escritorio\INFORMATICA\MATLAB\gato.png');

% 3) Detectar: devuelve cajas (bboxes), puntajes y etiquetas
[bboxes, scores, labels] = detect(detector, I);

% 4) Dibujar los resultados sobre la imagen
etiquetas = string(labels) + " " + string(round(scores,2));
Ianotada = insertObjectAnnotation(I, 'rectangle', bboxes, etiquetas, ...
    'LineWidth', 2, ...
    'FontSize', 50);

% 5) Mostrar
figure;
imshow(Ianotada);
title('Detección con tiny-YOLOv2 (COCO)');

% 6) Info en consola
fprintf('Objetos detectados: %d\n', size(bboxes,1));
disp(table(labels, scores));
