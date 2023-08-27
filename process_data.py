import os
import glob
import shutil
from PIL import Image
import numpy as np


#-----------------#
# label_folder = '/data2/chaoyi/HSI Dataset/V2/label'
# data_folder = '/data2/chaoyi/HSI Dataset/V2'

# # hsd_files = glob.glob(f"{data_folder}/*.hsd")
# # for hsd_file in hsd_files:
# #     new_folder = os.path.join(data_folder, hsd_file.split('.')[0])
# #     os.mkdir(new_folder)
# #     shutil.move(os.path.join(data_folder, hsd_file), new_folder)

# img_folder_list = os.listdir(data_folder)
# for img_folder in img_folder_list:
#     if img_folder == 'label':
#         continue
#     rgb_file = os.path.join(label_folder, 'rgb' + img_folder + '.jpg')
#     ground_truth = os.path.join(label_folder, 'rgb' + img_folder + '_gray' + '.png')
#     ground_truth_rgb = os.path.join(label_folder, 'rgb' + img_folder + '.png')
#     target_path = os.path.join(data_folder, img_folder)
    
#     target_rgb = os.path.join(target_path, 'rgb' + img_folder + '.jpg')
#     if os.path.exists(target_rgb):
#         os.remove(target_rgb)
#     target_ground_truth = os.path.join(target_path, 'rgb' + img_folder + '_gray' + '.png')
#     if os.path.exists(target_ground_truth):
#         os.remove(target_ground_truth)
#     target_ground_truth_rgb = os.path.join(target_path, 'rgb' + img_folder + '.png')
#     if os.path.exists(target_ground_truth_rgb):
#         os.remove(target_ground_truth_rgb)
#     shutil.move(rgb_file, target_path)
#     shutil.move(ground_truth, target_path)
#     shutil.move(ground_truth_rgb, target_path)
    
#-----------------#

new_data_folder = '/data2/chaoyi/HSI Dataset/V2/segmentai'
train_folder = '/data2/chaoyi/HSI Dataset/V2/train'
val_folder = '/data2/chaoyi/HSI Dataset/V2/test'

max_label = 0
for file in os.listdir(new_data_folder):
    if file.endswith(".jpg"):
        rgb_file = os.path.join(new_data_folder, file)
        
        ground_truth = file.split('.')[0] + '_grey' + '.png'
        ground_truth = os.path.join(new_data_folder, ground_truth)    
        ground_truth_array = np.array(Image.open(ground_truth))
        max_label = max(max_label, np.max(ground_truth_array))    
        if np.max(ground_truth_array) > 20:
            print(file, np.max(ground_truth_array))

imgs = []

for file in os.listdir(new_data_folder):
    if file.endswith(".jpg"):
        rgb_file = os.path.join(new_data_folder, file)
        
        ground_truth = file.split('.')[0] + '_grey' + '.png'
        ground_truth = os.path.join(new_data_folder, ground_truth)
        os.rename(ground_truth, ground_truth.replace('_grey', '_gray'))
        ground_truth = ground_truth.replace('_grey', '_gray')
        
        ground_truth_rgb = file.split('.')[0]  + '.png'
        ground_truth_rgb = os.path.join(new_data_folder, ground_truth_rgb)
        # os.rename(ground_truth_rgb, ground_truth_rgb.replace('_label_ground-truth_coco-panoptic', ''))
        # ground_truth_rgb = ground_truth_rgb.replace('_label_ground-truth_coco-panoptic', '')
        # ground_truth_rgb = ground_truth_rgb.replace('_label_ground-truth_coco-panoptic', '')
        
        file = file.split('.')[0].replace('rgb', '')
        
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
    