ward = 'C:\Users\mingyou.kwong\Google Drive\Universiti_Teknologi_Malaysia\Die_Defect_Classification'; 
cd (ward); 
addpath (ward); 

no_of_sample = 1; 
%load(fullfile(ward, 'matlab', rcnn3_5.mat')); 

roiDir = fullfile(ward, 'Dataset', 'ROI3_compared_5'); 
resultDir = fullfile(ward, 'Results\Classification3_5_2'); 
if ~exist(resultDir, 'dir') 
    mkdir(resultDir); 
end
addpath (resultDir); 

cd(roiDir); 
files = dir('*.png'); 
for j = 1:20
    I = files(j).name
    no_of_sample = no_of_sample + 1; 

image = fullfile('Dataset', 'ROI3_compared_5', I); 
img = imread(image);
[bbox, score, label] = detect(rcnn, img, 'MiniBatchSize', 10);
[score, idx] = max(score);

bbox = bbox(idx, :);

if (isempty(score))
    fprintf ('No defect detected\n'); 
else
    annotation = sprintf('%s: (Confidence = %f)', label(idx), score);
    detectedImg = insertObjectAnnotation(img, 'rectangle', bbox, annotation, 'FontSize', 40, 'LineWidth', 10);
%    figure;
%    imshow(detectedImg);
    cd (resultDir);
    imwrite(detectedImg, [I, '_', char(label(idx)), '.png'], 'png'); 
    cd(roiDir); 
end
end

