clear
clc

% load the matfile
filepath = '/data2/chaoyi/HSI_Dataset/HSI Drive/v1/cubes_float32/'; %%%%%%%%%%%%% full path to the main folder
list_mat = dir(fullfile(filepath,'*.mat'));

% load the filter
 % Single filter
EC_filter = load('EC_filter_20240205.mat'); %%%%%%%%%%%%%%%%%%   folder of the EC_filter
rawfilter = abs(EC_filter.respons);
rawfilter = rawfilter ./ max(rawfilter);
k = [40:15:400];
responsivity = rawfilter(k,:);
voltage_index = [1:1:71];

 % Dual filter
EC_filter2 = load('EC_filter_with_Trans_20240420.mat');  %%%%%%%%%%%%%%%%%%   folder of the EC_filter
rawfilter = abs(EC_filter2.responsivity);
rawfilter = rawfilter ./ max(rawfilter);
rawfilter2 = rawfilter(:,[1:2:503]);
k2 = [40:15:400];
responsivity2 = rawfilter2(k2,:);
voltage_index2 = [1:1:252];

[filepath,name,ext] = fileparts(fullfile(list_mat(1).folder,list_mat(1).name));
foldername = fullfile(filepath,"Sin_HVI");
if ~exist(foldername, 'dir')
    % Folder does not exist, so create it
    mkdir(foldername);
    fprintf('Folder created: %s\n', foldername);
else
    fprintf('Folder already exists: %s\n', foldername);
end
foldername = fullfile(filepath,"Dual_HVI");
if ~exist(foldername, 'dir')
    % Folder does not exist, so create it
    mkdir(foldername);
    fprintf('Folder created: %s\n', foldername);
else
    fprintf('Folder already exists: %s\n', foldername);
end

h = waitbar(0, 'Please wait...'); % create waitbar
for i = 1:length(list_mat)
    % collect data
    [filepath,name,ext] = fileparts(fullfile(list_mat(i).folder,list_mat(i).name));
    data1 = load(fullfile(list_mat(i).folder,list_mat(i).name));
    data2 = cell2mat(struct2cell(data1));
    data = permute(data2,[2 3 1]);
    clear data1
    clear data2

    % Filtering the image pixel by pixel;
    [x,y,z] = size(data);
    [c,r] = size(responsivity);
    [c2,r2] = size(responsivity2);
    filtered_img = zeros(x,y,r);
    filtered_img2 = zeros(x,y,r2);

    for ii = 1:x
        for jj = 1:y
            filtered_img(ii,jj,1:r) =  abs(responsivity)' * squeeze(data(ii,jj,:));
            filtered_img2(ii,jj,1:r2) =  abs(responsivity2)' * squeeze(data(ii,jj,:));
        end
    end
    clear data

    % Save the filtered data
    filename = fullfile(filepath,"Sin_HVI",[name,'.mat']);
    filename2 = fullfile(filepath,"Dual_HVI",[name,'.mat']);
    save(filename,"filtered_img","voltage_index"); % save as the .mat file
    filtered_img = filtered_img2;
    save(filename2,"filtered_img","voltage_index2"); % save as the .mat file
    waitbar(i/length(list_mat), h, sprintf('Progress: %d%%', round(100*i/length(list_mat)))); % update waitbar
end    
close(h); % close waitbar