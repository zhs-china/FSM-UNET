import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class folder_loader(Dataset):

    def __init__(self, root_dir, size=(256, 256)):
        super().__init__()
        self.root_dir = root_dir
        self.size = size
        self.image_dir = os.path.join(root_dir, 'image')
        self.label_dir = os.path.join(root_dir, 'label')

        self.filenames = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img_path = os.path.join(self.image_dir, fname)
        label_fname = os.path.splitext(fname)[0]
        label_path = None
        for ext in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'):
            cand = os.path.join(self.label_dir, label_fname + ext)
            if os.path.exists(cand):
                label_path = cand
                break
        if label_path is None:
            label_path = os.path.join(self.label_dir, fname)

        img = Image.open(img_path).convert('RGB').resize(self.size, Image.BILINEAR)
        img_np = np.array(img, dtype=np.float32)

        label = Image.open(label_path).convert('L').resize(self.size, Image.NEAREST)
        label_np = np.array(label, dtype=np.float32)
        label_np = (label_np > 127).astype(np.float32)

        img_t = torch.from_numpy(img_np).permute(2, 0, 1) / 255.0
        seg_t = torch.from_numpy(label_np).unsqueeze(0)

        return img_t, seg_t
