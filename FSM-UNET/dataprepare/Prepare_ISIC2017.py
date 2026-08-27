# -*- coding: utf-8 -*-

import h5py
import numpy as np
import scipy.io as sio
from PIL import Image
import glob
import os

height = 256
width = 256
channels = 3

current_dir = os.getcwd()
print(f"Current working directory: {current_dir}")

Dataset_add = os.path.join(os.path.dirname(current_dir), 'data', 'dataset_isic17')
Tr_add = 'ISIC-2017_Training_Data'

full_path = os.path.join(Dataset_add, Tr_add, '*.jpg')
print(f"Searching for images at path: {full_path}")

Tr_list = glob.glob(full_path)
print(f"Total images found: {len(Tr_list)}")

if len(Tr_list) == 0:
    dir_path = os.path.join(Dataset_add, Tr_add)
    print(f"Directory exists: {os.path.exists(dir_path)}")
    if os.path.exists(dir_path):
        print(f"Directory contents: {os.listdir(dir_path)}")

Data_train_2017 = np.zeros([2000, height, width, channels])
Label_train_2017 = np.zeros([2000, height, width])

print('Reading ISIC 2017')
for idx in range(len(Tr_list)):
    print(f"Processing image {idx + 1}: {Tr_list[idx]}")
    img = Image.open(Tr_list[idx])
    img = img.resize((width, height), Image.BILINEAR)
    img = np.array(img)
    Data_train_2017[idx, :, :, :] = img

    b = Tr_list[idx]
    a = b[0:len(Dataset_add)]
    b = b[len(b) - 16: len(b) - 4]
    add = os.path.join(a, 'ISIC2017_Task1_Training_GroundTruth', f"{b}_segmentation.png")
    print(f"Looking for mask at: {add}")
    img2 = Image.open(add)
    img2 = img2.resize((width, height), Image.BILINEAR)
    img2 = np.array(img2)
    Label_train_2017[idx, :, :] = img2

print('Reading ISIC 2017 finished')

Train_img = Data_train_2017[0:1250, :, :, :]
Validation_img = Data_train_2017[1250:1250 + 150, :, :, :]
Test_img = Data_train_2017[1250 + 150:2000, :, :, :]

Train_mask = Label_train_2017[0:1250, :, :]
Validation_mask = Label_train_2017[1250:1250 + 150, :, :]
Test_mask = Label_train_2017[1250 + 150:2000, :, :]

np.save('data_train_isic17.npy', Train_img)
np.save('data_test_isic17.npy', Test_img)
np.save('data_val_isic17', Validation_img)

np.save('mask_train_isic17.npy', Train_mask)
np.save('mask_test_isic17.npy' , Test_mask)
np.save('mask_val_isic17.npy'  , Validation_mask)
