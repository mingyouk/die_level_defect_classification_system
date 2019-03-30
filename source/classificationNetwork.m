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
defectDatasetPath = fullfile(ward, 'Dataset', 'Classification3_3'); 
%defectData = imageDatastore(defectDatasetPath,...
%        'IncludeSubfolders',true,'LabelSource','foldernames');
category = [...
    string('Blob'), ...
    string('DieCrack'), ...
    string('PinHole'), ...
    string('Underfill')...
    ];  
addpath(defectDatasetPath); 
%numClasses = numel(categories(defectData.Labels)) + 1;
numClasses = numel(category) + 1;
%load('C:\Users\mingyou.kwong\Google Drive\Universiti_Teknologi_Malaysia\Die_Defect_Classification\matlab\rcnn3_2.mat'); 

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
  'MaxEpochs', 10);

%rcnn = trainRCNNObjectDetector(classification_3_6, layers, options, 'NegativeOverlapRange', [0 0.3]);
rcnn = trainRCNNObjectDetector(classification_3_6, rcnn.Network.Layers, options, 'NegativeOverlapRange', [0 0.3]);
save ('matlab\rcnn3_5_new.mat', 'rcnn'); 

img = imread('Dataset\ROI3_compared_5\Blob_01_1.png');
[bbox, score, label] = detect(rcnn, img, 'MiniBatchSize', 10);
[score, idx] = max(score);

bbox = bbox(idx, :);
annotation = sprintf('%s: (Confidence = %f)', label(idx), score);

detectedImg = insertObjectAnnotation(img, 'rectangle', bbox, annotation);

figure
imshow(detectedImg)