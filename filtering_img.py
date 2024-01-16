import os
import glob
import shutil
from PIL import Image
import numpy as np


def save_filtered_folder_path(data_folder_path):
    '''
    For the HSI dataset, "data_folder" is the path to the train or test folder
    read the "gray" label image and if it contains car and building, 
    and has less than 5 different labels, then append the folder path to the list
    save the list to a txt file
    '''
    data_folders = os.listdir(data_folder_path)
    filtered_folders = []
    for data_folder in data_folders:
        label_path = os.path.join(data_folder_path, data_folder, 'rgb' + data_folder + '_gray.png')
        label = np.array(Image.open(label_path))
        if len(np.unique(label)) < 8 and (label == 13).any() and (label == 2).any():
            filtered_folders.append(os.path.join(data_folder_path, data_folder).replace('/data2/chaoyi/HSI_Dataset/V2/', ''))
    filtered_folders.sort()
    return filtered_folders


if __name__ == "__main__":
    data_folder_path = ['/data2/chaoyi/HSI_Dataset/V2/train/', '/data2/chaoyi/HSI_Dataset/V2/test/']
    filtered_folders = []
    for data_folder in data_folder_path:
        filtered_folders += save_filtered_folder_path(data_folder)
    print(len(filtered_folders))
    filtered_folders.sort()
    with open('/data2/chaoyi/HSI_Dataset/V2/filtered_folders.txt', 'w') as f:
        for filtered_folder in filtered_folders:
            f.write(filtered_folder + '\n')