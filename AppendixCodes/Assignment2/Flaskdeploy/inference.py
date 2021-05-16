# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 00:49:27 2021

@author: guang
"""
import torch
from torchvision import datasets, models, transforms
from PIL import Image
import pandas as pd
import io

modelPath = './cleaned4class_100epoch_false'
model = torch.load(modelPath,map_location='cpu')

def Predict(image):
    codes = {'cake':0, 'pasta':1, 'pizza':2, 'drink':3}
    tag_pred = pd.Series(codes)
    data_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    # inputs = Image.open(io.BytesIO(image)).convert('RGB')
    inputs = Image.open(image).convert('RGB')
    inputs = data_transforms(inputs).unsqueeze(0)
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    title = tag_pred.index[preds]
    return title