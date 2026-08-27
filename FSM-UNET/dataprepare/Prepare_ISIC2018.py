# -*- coding: utf-8 -*-

import h5py
import numpy as np
import scipy.io as sio
from PIL import Image
import glob
import os

height = 256
width  = 256
channels = 3

project_root = os.path.dirname(os.getcwd())
Dataset_add = os.path.join(project_root, 'data', 'dataset_isic18')
Tr_add = 'ISIC2018_Task1-2_Training_Input'

search_path = os.path.join(Dataset_add, Tr_add, '*.jpg')
print(f"Searching for images in: {search_path}")
Tr_list = glob.glob(search_path)
print(f"Found {len(Tr_list)} images.")

Data_train_2018    = np.zeros([2594, height, width, channels])
Label_train_2018   = np.zeros([2594, height, width])

print('Reading ISIC 2018')
if not Tr_list:
    print("Warning: No images found. Please check the dataset path and contents.")
else:
    for idx in range(len(Tr_list)):
        print(f"Processing {idx+1}/{len(Tr_list)}: {os.path.basename(Tr_list[idx])}")
        img = Image.open(Tr_list[idx])
        img = img.resize((width, height), Image.BILINEAR)
        Data_train_2018[idx, :,:,:] = np.array(img)

        img_basename = os.path.basename(Tr_list[idx])
        img_name = os.path.splitext(img_basename)[0]

        mask_path = os.path.join(Dataset_add, 'ISIC2018_Task1_Training_GroundTruth', f"{img_name}_segmentation.png")

        try:
            img2 = Image.open(mask_path)
            img2 = img2.resize((width, height), Image.BILINEAR)
            Label_train_2018[idx, :,:] = np.array(img2)
        except FileNotFoundError:
            print(f"  - Warning: Mask not found for {img_basename} at {mask_path}")

print('Reading ISIC 2018 finished')

Train_img      = Data_train_2018[0:1815,:,:,:]
Validation_img = Data_train_2018[1815:1815+259,:,:,:]
Test_img       = Data_train_2018[1815+259:2594,:,:,:]

Train_mask      = Label_train_2018[0:1815,:,:]
Validation_mask = Label_train_2018[1815:1815+259,:,:]
Test_mask       = Label_train_2018[1815+259:2594,:,:]


np.save('data_train_isic18.npy', Train_img)
np.save('data_test_isic18.npy' , Test_img)
np.save('data_val_isic18.npy'  , Validation_img)

np.save('mask_train_isic18.npy', Train_mask)
np.save('mask_test_isic18.npy' , Test_mask)
np.save('mask_val_isic18.npy'  , Validation_mask)
