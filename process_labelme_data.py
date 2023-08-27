import os
import glob
import shutil
from PIL import Image
import numpy as np


new_data_folder = '/data2/chaoyi/HSI Dataset/segments/RexKim150imagesDone'
train_folder = '/data2/chaoyi/HSI Dataset/V2/train'
val_folder = '/data2/chaoyi/HSI Dataset/V2/val'

imgs = []
img_folder_list = os.listdir(new_data_folder)

for img_folder in img_folder_list:
    file = img_folder
    rgb_file = os.path.join(new_data_folder, img_folder, 'img.png')
    
    ground_truth = os.path.join(new_data_folder, img_folder, 'label.png')
    os.rename(ground_truth, ground_truth.replace('label', file + '_gray'))
    ground_truth = ground_truth.replace('label', file + '_gray')
    
    ground_truth_rgb = os.path.join(new_data_folder, img_folder, 'label_viz.png')
    os.rename(ground_truth_rgb, ground_truth_rgb.replace('label_viz', file))
    ground_truth_rgb = ground_truth_rgb.replace('label_viz', file)
    
    png_image = Image.open(rgb_file)
    rgb_file = rgb_file.replace('png', 'jpg')
    if png_image.mode != 'RGB':
        png_image = png_image.convert('RGB')
    png_image.save(rgb_file, 'JPEG')
    png_image.close()
    
    file = file.replace('rgb', '')
    
    imgs.append([file, rgb_file, ground_truth, ground_truth_rgb])

folder_name_dict = {}
img_folder_list = os.listdir(train_folder) + os.listdir(val_folder)
for i, img_folder in enumerate(img_folder_list):
    folder_name_dict[img_folder] = os.path.join(train_folder, img_folder) \
        if i < len(os.listdir(train_folder)) else os.path.join(val_folder, img_folder)

for img in imgs:
    file = img[0]
    rgb_file = img[1]
    ground_truth = img[2]
    ground_truth_rgb = img[3]
    
    folder_name = folder_name_dict[file]
    
    target_path = os.path.join(folder_name, os.path.basename(ground_truth))
    if os.path.exists(target_path):
        os.remove(target_path)
    shutil.move(ground_truth, folder_name)
    
    target_path = os.path.join(folder_name, os.path.basename(ground_truth_rgb))
    if os.path.exists(target_path):
        os.remove(target_path)
    shutil.move(ground_truth_rgb, folder_name)