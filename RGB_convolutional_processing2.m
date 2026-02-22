clear
filepath  = 'E:\RGB_image_processing';
rgbImage = imread('T4.png'); %'apple.jpg' 'confocal-image.jpg' 'peppers.png' Lenna.png hsi_Flower.jpg color_f1.png color_f2.png stock-photo-multicolored-flower
allBlack = zeros(size(rgbImage, 1), size(rgbImage, 2), 'uint8');

%% RGB Wight
filename = 'filtered_rgbWight.mat';
Wights = cell2mat(struct2cell(load(filename)));
RGB_Wights = Wights(:,:,1);
[c,r] = size(RGB_Wights);
if c > 40
    RGB_Wights = vertcat( -RGB_Wights(1:30,:), RGB_Wights(31:end,:) );
else
    RGB_Wights = vertcat( -RGB_Wights(1:15,:), RGB_Wights ...
        ...
        (16:end,:) );
end
RGB_Wights(abs(RGB_Wights)<0.08) = 0;
[c,r] = size(RGB_Wights);
%% 
% for i = 1:c

% rwight = abs(RGB_Wights(i,1));
% gwight = abs(RGB_Wights(i,2));
% bwight = abs(RGB_Wights(i,3));
% 
% rgb = [rwight,gwight,bwight];
% rwight = rwight/max(rgb);
% gwight = gwight/max(rgb);
% bwight = bwight/max(rgb);

%%
% Creat picture 
figure(2);clf(2);figure(2);
[ha, pos] = tight_subplot(4, 2, [.005 .005],[.1 .01],[.01 .01]);

set(figure(2),'Position',[600 50 460 900]);
% subplot(2, 4, 4);
axes(ha(1)); 
imshow(rgbImage);

% Extract the individual red, green, and blue color channels.
redChannel = rgbImage(:, :, 1);
% redimage = subplot(2, 4, 1);
axes(ha(3)); 
RImage = cat(3, redChannel, allBlack, allBlack);
imshow(RImage);
% colormap(redimage,othercolor('Reds9'));

greenChannel = rgbImage(:, :, 2);
% greenimage = subplot(2, 4, 2);
axes(ha(5)); 
GImage = cat(3, allBlack, greenChannel, allBlack);
imshow(GImage);
% colormap(greenimage, othercolor('Greens9'));   

blueChannel = rgbImage(:, :, 3);
% buleimage = subplot(2, 4, 3);
axes(ha(7)); 
BImage = cat(3, allBlack, allBlack, blueChannel);
imshow(BImage);
% colormap(buleimage, othercolor('Blues9')); 

%% My Kernels

% My Kernel, RGB kernel extracted from DATA
mykernelR = [0.6206,0.6206,0.6206; 0,0,0; -0.4244,-0.4244,-0.4244];
mykernelG = [0.41,0.41,0.41; 0,0,0; -0.40,-0.40,-0.40] ;
mykernelB = [0,0,0; 0,0,0; 0,0,0] ;

%% Image processing

% Red processing
kr = 1;
% filteredredimage = subplot(2, 4, 5);
axes(ha(4)); 
filteredredChannel = redChannel .* kr;
filteredredChannel = conv2(filteredredChannel,mykernelR, 'same');
fRImage = cat(3, filteredredChannel, allBlack, allBlack);
imshow(fRImage);
% colormap(filteredredimage,othercolor('Reds9'));

% Green processing
kg = 1;
% filteredgreenimage = subplot(2, 4, 6);
axes(ha(6));
filteredgreenChannel = greenChannel .* kg;
filteredgreenChannel = conv2(filteredgreenChannel,mykernelG, 'same');
fGImage = cat(3, allBlack, filteredgreenChannel, allBlack);
imshow(fGImage);
% colormap(filteredgreenimage, othercolor('Greens9'));

% Blue processing
kb = 1;
% filteredblueimage = subplot(2, 4, 7);
axes(ha(8));
filteredblueChannel = blueChannel .* kb;
% filteredblueChannel = imgaussfilt(blueChannel,4);
filteredblueChannel = conv2(filteredblueChannel,mykernelB, 'same');
fGImage = cat(3, allBlack, allBlack, filteredblueChannel);
imshow(fGImage);
% colormap(filteredblueimage, othercolor('Blues9')); 

% Recombine separate color channels into a single, true color RGB image.
rgbImage2 = cat(3, uint8(filteredredChannel), uint8(filteredgreenChannel), uint8(filteredblueChannel));
axes(ha(2));
% subplot(2, 4, 8);
imshow(rgbImage2);
% saveas(2,['filtered_rgb_',num2str(i),'.jpg'])

% end