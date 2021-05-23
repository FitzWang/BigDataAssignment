# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 23:57:53 2021

@author: guang
"""
import os
import torch
from torch.utils.data import Dataset
from PIL import Image


class Dataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = annotations_file
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        try:
            img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0]+'.png')
            image = Image.open(img_path).convert('RGB')
        except IOError:
            idx = idx+1
            img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0]+'.png')
            image = Image.open(img_path).convert('RGB')
        # image = read_image(img_path)
        label = torch.tensor(self.img_labels.iloc[idx, -1])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        # sample = {"image": image, "label": label}
        # return sample
        return image,label