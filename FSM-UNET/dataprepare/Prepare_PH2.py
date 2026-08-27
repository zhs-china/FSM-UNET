# -*- coding: utf-8 -*-

import h5py
import numpy as np
import scipy.io as sio
import glob
import os
from PIL import Image

height = 256
width  = 256
channels = 3

DATASET_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'PH2')


if not DATASET_ROOT:
    raise ValueError('请在 Prepare_PH2.py 顶部设置 DATASET_ROOT 为 PH2 根目录的绝对路径')

Dataset_add = os.path.abspath(DATASET_ROOT)
image_dir = os.path.join(Dataset_add, 'images')
mask_dir = os.path.join(Dataset_add, 'masks')

print(f"cwd: {os.getcwd()}")
print(f"script: {os.path.abspath(__file__)}")
print(f"DATASET_ROOT: {Dataset_add}")
print(f"images dir: {image_dir}")
print(f"masks dir:  {mask_dir}")

img_patterns = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG', '*.bmp', '*.BMP']
Tr_list = []
for pat in img_patterns:
    Tr_list.extend(glob.glob(os.path.join(image_dir, pat)))
Tr_list = sorted(list(set(Tr_list)))

if len(Tr_list) == 0:
    raise FileNotFoundError(f"No images found in: {os.path.abspath(image_dir)}; supported: {img_patterns}")
n_samples = len(Tr_list)
Data_all    = np.zeros([n_samples, height, width, channels])
Label_all   = np.zeros([n_samples, height, width])

print('Reading PH2')
print(Tr_list)
for idx in range(len(Tr_list)):
    print(idx+1)
    img = Image.open(Tr_list[idx]).convert('RGB').resize((width, height), Image.BILINEAR)
    img = np.asarray(img, dtype=np.float64)
    Data_all[idx, :,:,:] = img

    img_path = Tr_list[idx]
    base = os.path.splitext(os.path.basename(img_path))[0]
    exts = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG', '.bmp', '.BMP']
    mask_candidates = [os.path.join(mask_dir, base + ext) for ext in exts]
    if not any(os.path.exists(p) for p in mask_candidates):
        wildcards = []
        for ext in exts:
            wildcards.extend(glob.glob(os.path.join(mask_dir, base + '*' + ext)))
        def _score(p):
            name = os.path.basename(p).lower()
            score = 0
            if 'lesion' in name: score += 3
            if 'mask' in name:   score += 2
            if 'seg' in name:    score += 1
            return (-score, len(name))
        wildcards = sorted(set(wildcards), key=_score)
        mask_path = wildcards[0] if len(wildcards) > 0 else None
        if mask_path is None:
            raise FileNotFoundError(f"Mask not found for: {img_path}. Tried exact: {mask_candidates}; wildcard none found under {mask_dir}")
    else:
        mask_path = next((p for p in mask_candidates if os.path.exists(p)), None)
    print(f"match mask: {os.path.basename(mask_path)} for image: {os.path.basename(img_path)}")
    m = Image.open(mask_path).convert('L').resize((width, height), Image.NEAREST)
    img2 = np.asarray(m, dtype=np.float64)
    Label_all[idx, :,:] = img2


print('Reading PH2 finished')

np.random.seed(42)
N = Data_all.shape[0]
indices = np.arange(N)
np.random.shuffle(indices)

n_train = int(N * 0.7)
n_val = int(N * 0.1)
n_test = N - n_train - n_val

train_idx = indices[:n_train]
val_idx = indices[n_train:n_train + n_val]
test_idx = indices[n_train + n_val:]

Train_img = Data_all[train_idx, :, :, :]
Validation_img = Data_all[val_idx, :, :, :]
Test_img = Data_all[test_idx, :, :, :]

Train_mask = Label_all[train_idx, :, :]
Validation_mask = Label_all[val_idx, :, :]
Test_mask = Label_all[test_idx, :, :]

np.save('data_train_PH2.npy', Train_img)
np.save('data_val_PH2.npy', Validation_img)
np.save('data_test_PH2.npy' , Test_img)

np.save('mask_train_PH2.npy', Train_mask)
np.save('mask_val_PH2.npy'  , Validation_mask)
np.save('mask_test_PH2.npy' , Test_mask)

print(f"PH2 split counts -> train: {n_train}, val: {n_val}, test: {n_test}")
print(f"Saved: data_train_PH2.npy{Train_img.shape}, data_val_PH2.npy{Validation_img.shape}, data_test_PH2.npy{Test_img.shape}")
print(f"Saved: mask_train_PH2.npy{Train_mask.shape}, mask_val_PH2.npy{Validation_mask.shape}, mask_test_PH2.npy{Test_mask.shape}")
