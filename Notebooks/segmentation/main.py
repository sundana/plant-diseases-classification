import cv2
import numpy as np
from tqdm import tqdm
import os

source_path = '/Users/firmansyahsundana/Documents/research/computer_science/plant-diseases-classification/Data/External/Corn/Corn_(maize)___Northern_Leaf_Blight'
mask_path = '/Users/firmansyahsundana/Documents/research/computer_science/plant-diseases-classification/Pytorch-UNet/output/Corn_(maize)___Northern_Leaf_Blight'
output_path = '/Users/firmansyahsundana/Documents/research/computer_science/plant-diseases-classification/Data/Processed/segmented_images/Corn_(maize)___Northern_Leaf_Blight'

def segmentation(src_dir, mask_dir, output_dir):
    for filename in tqdm(os.listdir(source_path)):
        ori_img = cv2.imread(source_path + '/' + filename, cv2.COLOR_BGR2RGB)
        mask_path = mask_dir + '/' + filename.replace('.jpg','').replace('.JPG', '') + '_mask.jpg'
        # print(mask_path)
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        _, binary_mask = cv2.threshold(mask_img, 128, 255, cv2.THRESH_BINARY)
        segmented = cv2.bitwise_and(ori_img,ori_img, mask=binary_mask)
        cv2.imwrite(output_dir + '/' + filename, segmented)
        


segmentation(source_path, mask_path, output_path)