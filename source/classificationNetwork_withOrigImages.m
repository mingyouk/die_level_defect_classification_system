% Get GPU device information
deviceInfo = gpuDevice;

% Check the GPU compute capability
computeCapability = str2double(deviceInfo.ComputeCapability);
assert(computeCapability >= 3.0, ...
    'This example requires a GPU device with compute capability 3.0 or higher.'); 

% Set variables
ward = 'C:\Users\mingyou.kwong\Google Drive\Universiti_Teknologi_Malaysia\Die_Defect_Classification'; 
addpath(ward); 
cd(ward); 

% Training dataset
defectDatasetPath = fullfile(ward, 'Dataset', 'Original'); 
%defectData = imageDatastore(defectDatasetPath,...
%        'IncludeSubfolders',true,'LabelSource','foldernames');
category = [...
    string('Blob'), ...
    string('PinHole'), ...
    string('Underfill')...
    ];  
addpath(defectDatasetPath); 
numClasses = numel(category) + 1;

% Network
net = alexnet; 
layersTransfer = net.Layers(1:end-3);

layers = [...
    layersTransfer
    fullyConnectedLayer(numClasses) 
    softmaxLayer
    classificationLayer];
 
options = trainingOptions('sgdm', ...
  'MiniBatchSize', 15, ...
  'InitialLearnRate', 1e-6, ...
  'MaxEpochs', 100);


%rcnn = trainRCNNObjectDetector(DieLevelDefect, layers, options, 'NegativeOverlapRange', [0 0.1]);
%save ('matlab\rcnn7_2.mat', 'rcnn');

fprintf ('Testing network\n');

img = imread('Dataset\ROI5_withROI_copy\Blob_10.png');
[bbox, score, label] = detect(rcnn, img, 'MiniBatchSize', 15);
[score1, idx] = max(score);

bbox1 = bbox(idx, :);
img2 = img; 

if (isempty(score1))
    fprintf ('Oh no! No defect detected\n'); 
    
else
    detectedImg = img; 
    for i = size(score)
    annotation = sprintf('%s: (Confidence = %f)', label(i), score(i));
    detectedImg = insertObjectAnnotation(detectedImg, 'rectangle', bbox(i,:), annotation, 'FontSize', 40, 'LineWidth', 10);
    end
        figure;
    imshow(detectedImg);
%    cd (resultDir);
%    imwrite(detectedImg, [I, '_', char(label(idx)), '.png'], 'png'); 
%    cd(roiDir); 

end