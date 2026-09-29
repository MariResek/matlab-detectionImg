I = imread('C:\Users\mrese\OneDrive\Escritorio\INFORMATICA\MATLAB\gato.png'); 
[bboxes, scores, labels] = yolov2_detect(I, 'yoloDetector.mat'); 


Iout = insertObjectAnnotation(I, ... 
    "rectangle", ... 
    bboxes, ... 
    labels, ...
    'FontSize', 50); 

imshow(Iout);



%con este cacho de codigo tira todas las class
%load('yoloDetector.mat');

%for i = 1:numel(detector.ClassNames)
    %fprintf('%d - %s\n', i, detector.ClassNames(i));
    %end