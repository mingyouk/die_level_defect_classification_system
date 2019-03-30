ward = 'C:\Users\mingyou.kwong\Google Drive\Universiti_Teknologi_Malaysia\Die_Defect_Classification'; 
cd (ward); 
addpath (ward); 

no_of_sample = 1; 
%load(fullfile(ward, 'matlab', rcnn3_5.mat')); 

roiDir = fullfile(ward, 'Dataset', 'ROI5_withROI_copy'); 
resultDir = fullfile(ward, 'Results\Classification_7_3'); 
if ~exist(resultDir, 'dir') 
    mkdir(resultDir); 
end
addpath (resultDir); 

cd(roiDir); 
files = dir('*.png'); 
for j = 1:length(files)
    I = files(j).name
    no_of_sample = no_of_sample + 1; 

image = fullfile(roiDir, I); 
img = imread(image);
[bbox, score, label] = detect(rcnn, img, 'MiniBatchSize', 15);
[score, idx] = max(score);

bbox = bbox(idx, :);

if (isempty(score))
    cd (resultDir);
    imwrite(img, [I, '_NoDefectDetected', '.png'], 'png'); 
    cd(roiDir);
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

