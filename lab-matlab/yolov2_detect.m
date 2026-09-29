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