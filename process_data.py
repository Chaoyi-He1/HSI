import os
import glob
import shutil


new_data_folder = '/data2/chaoyi/HSI Dataset/segmentai'
train_folder = '/data2/chaoyi/HSI Dataset/V2/train'
val_folder = '/data2/chaoyi/HSI Dataset/V2/val'

imgs = []

for file in os.listdir(new_data_folder):
    if file.endswith(".jpg"):
        rgb_file = os.path.join(new_data_folder, file)
        
        ground_truth = file.split('.')[0] + '_label_ground-truth' + '.png'
        ground_truth = os.path.join(new_data_folder, ground_truth)
        # os.rename(ground_truth, ground_truth.replace('_grey', '_gray'))
        ground_truth = ground_truth.replace('_label_ground-truth', '_gray')
        
        ground_truth_rgb = file.split('.')[0] + '_label_ground-truth_coco-panoptic' + '.png'
        ground_truth_rgb = os.path.join(new_data_folder, ground_truth_rgb)
        # os.rename(ground_truth_rgb, ground_truth_rgb.replace('_label_ground-truth_coco-panoptic', ''))
        ground_truth_rgb = ground_truth_rgb.replace('_label_ground-truth_coco-panoptic', '')
        
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
    