clear;
clc;
loadPlotStyle_ran;

% load the image
filename = '20190528_180947_7718.hsd';
labelname = 'label_gray.png';
refldata = load('Tree_reflectance.mat');
reflect = refldata.reflect;

% Compute the wavelength of the Hyperspectral data
wavelength = [446:4:800];
%% extract ClassIMAGE hcube

image = imread(labelname);
[data1] = readHSD(filename);
data = abs(single(data1(:,:,1:89)));
clear data1

[row,col] = find(image==6); % tree class 6/sky 8 /road 3 /butting 7/car 1
HSimage = zeros(size(data));
z = length(HSimage(1,1,:));

for i= 1:length(row)
    HSimage(row(i),col(i),1:z) = data(row(i),col(i),1:z);
end
clear data
hcube = hypercube(HSimage,wavelength);

%% 

queryEndmember = [1];
threshold = 15;
rgbImg = colorize(hcube,'ContrastStretching',true);
overlayImg = rgbImg;
labelColor = {'Blue'};
segmentedImg = cell(size(hcube.DataCube,1),size(hcube.DataCube,2),numel(queryEndmember));
overlay = zeros(size(segmentedImg));

for num = 1:numel(queryEndmember)
    scoreMap = sid(hcube,reflect);
    segmentedImg{num} =  scoreMap > 0 & scoreMap <= threshold;
    index = scoreMap > 0 & scoreMap <= threshold;
    overlayImg = imoverlay(overlayImg,segmentedImg{num},labelColor{num});   
end
overlay(index) = 1;

figure('Position',[0 0 900 400])
subplot('Position',[0 0.2 0.3 0.7])
imagesc(rgbImg)
title('RGB Transformation of Test Data');
axis off
subplot('Position',[0.35 0.2 0.3 0.7])
imagesc(overlayImg)
title('Overlay Segmented Regions')
hold on
dim = [0.66 0.6 0.3 0.3];
annotation('textbox',dim,'String','Tree','Color',[1 1 1],'BackgroundColor',[0 0 1],'FitBoxToText','on');
hold off
axis off