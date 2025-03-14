clear
D = 'E:\training dataset'; %%%%%%%%%%%%% full path to the main folder
S = dir(fullfile(D,'*'));
N = setdiff({S([S.isdir]).name},{'.','..'}); % list of subfolders

% Load the EC Filter (10 voltages)
EC_filter = load('E:\RGB_image_processing\EC_filter.mat');  %%%%%%%%%%%%%%%%%%   folder of the EC_filter
band = load('E:\RGB_image_processing\informativeband.mat');
voltage_index = band.informativeband(1:10);
responsivity = EC_filter.responsivity(:,voltage_index);

% Modified and transfer the hsd data in each subfolders
for i = 1:numel(N)
    % Load the hsd file
    H = dir(fullfile(D,N{i},'*.hsd')); 
    [filepath,name,ext] = fileparts(H.name); 
    [data] = readHSD(H.name);

    % Compute the wavelength of the Hyperspectral data
    startw = 446;
    stepw = 4;
    endw = 958;
    wavelength = [startw:stepw:endw];

    % Creat hcube flie
    hcube = hypercube(data,wavelength);

    % Cut the wavelength of Hypespectral data to 400~800 nm
    wlrange = [800.1, max(wavelength) + 0.1];
    hcube = removeBands(hcube,'Wavelength',wlrange);

    % save the modified Hypespectral data
    data = hcube.DataCube;
    wavelength = hcube.Wavelength;
%     filename = fullfile(D,N{i},[name,'.mat']);
%     save(filename,"data","wavelength"); % save as the .mat file

    % Filtering the image pixel by pixel;
    [x,y,z] = size(hcube.DataCube);
    [c,r] = size(responsivity);
    filtered_img = zeros(x,y,r);
    
    for ii = 1:x
        for jj = 1:y
            filtered_img(ii,jj,1:r) =  abs(responsivity)' * squeeze(hcube.DataCube(ii,jj,:));
        end
    end

    % Save the filtered data
    filename = fullfile(D,N{i},[name,'_10channel.mat']);
    save(filename,"filtered_img","voltage_index"); % save as the .mat file
       
end