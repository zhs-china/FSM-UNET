# -*- coding: utf-8 -*-

import h5py
import numpy as np
import scipy.io as sio
import scipy.misc as sc
import glob

height = 256
width  = 256
channels = 3

train_number = 1000
val_number = 200
test_number = 400
all = int(train_number) + int(val_number) + int(test_number)

Tr_list = glob.glob("images"+'/*.png')
Data_train_2018    = np.zeros([all, height, width, channels])
Label_train_2018   = np.zeros([all, height, width])

print('Reading')
print(len(Tr_list))
for idx in range(len(Tr_list)):
    print(idx+1)
    img = sc.imread(Tr_list[idx])
    img = np.double(sc.imresize(img, [height, width, channels], interp='bilinear', mode = 'RGB'))
    Data_train_2018[idx, :,:,:] = img

    b = Tr_list[idx]
    b = b[len(b)-8: len(b)-4]
    add = ("masks/" + b +'.png')
    img2 = sc.imread(add)
    img2 = np.double(sc.imresize(img2, [height, width], interp='bilinear'))
    Label_train_2018[idx, :,:] = img2

print('Reading your dataset finished')

Train_img      = Data_train_2018[0:train_number,:,:,:]
Validation_img = Data_train_2018[train_number:train_number+val_number,:,:,:]
Test_img       = Data_train_2018[train_number+val_number:all,:,:,:]

Train_mask      = Label_train_2018[0:train_number,:,:]
Validation_mask = Label_train_2018[train_number:train_number+val_number,:,:]
Test_mask       = Label_train_2018[train_number+val_number:all,:,:]


np.save('data_train', Train_img)
np.save('data_test' , Test_img)
np.save('data_val'  , Validation_img)

np.save('mask_train', Train_mask)
np.save('mask_test' , Test_mask)
np.save('mask_val'  , Validation_mask)
