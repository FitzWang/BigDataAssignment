# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 08:40:49 2021

@author: guang
"""
import pandas as pd
from difflib import SequenceMatcher
import numpy as np
import MyImageDataset
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.optim as optim
import time
import copy
from sklearn.model_selection import train_test_split
os.environ['KMP_DUPLICATE_LIB_OK']='True'


global tag_pred

def Preprocesse():
    recipes = pd.read_csv("recipes.csv",sep=';')
    tags = [names for names in recipes.columns if names.startswith('tag_')]
    tagCounts = recipes.loc[:,tags].sum()
    # tagNames = pd.DataFrame({'tag':[name.split('_')[1] for name in tagCounts.index.to_list()]}).sort_values('tag')
    # matrix = cdist(tagNames.reshape(-1, 1), tagNames.reshape(-1, 1), lambda x, y: ratio(x[0], y[0]))
    # df = pd.DataFrame(data=matrix, index=tagNames, columns=tagNames)
    ## merge tags
    recipes_new = recipes.loc[:,['photo_id']]
    recipes_new['bake'] = recipes['tag_bake'] + recipes['tag_baker'] + recipes['tag_bakery'] + recipes['tag_baking']\
        + recipes['tag_cake'] + recipes['tag_cakes'] + recipes['tag_cheesecake'] + recipes['tag_pastry'] + recipes['tag_bread'] \
        + recipes['tag_cookies'] 
    recipes_new['dessert'] = recipes['tag_dessert'] + recipes['tag_desserts'] + recipes['tag_sweet'] + recipes['tag_sweets']\
        + recipes['tag_sweettooth'] + recipes['tag_icecream']
    recipes_new['vegan'] = recipes['tag_vegan'] + recipes['tag_veganfood'] + recipes['tag_veganrecipe'] + recipes['tag_vegetables']\
        + recipes['tag_vegetarian'] + recipes['tag_vegetarianrecipes'] + recipes['tag_veggie'] + recipes['tag_veggies'] + recipes['tag_plantbased']
    recipes_new['chocolate'] = recipes['tag_chocolate'] + recipes['tag_chocolat']
    # recipes_new['egg'] = recipes['tag_egg'] + recipes['tag_eggs']    
    recipes_new['fruit'] = recipes['tag_fruit'] + recipes['tag_lemon'] + recipes['tag_banana'] + recipes['tag_strawberry']
    recipes_new['italian'] = recipes['tag_italia'] + recipes['tag_italian'] + recipes['tag_italianfood'] + recipes['tag_italy'] \
        + recipes['tag_pasta'] + recipes['tag_pastalover'] + recipes['tag_pizza']
    recipes_new['meat'] = recipes['tag_chicken'] + recipes['tag_protein'] + recipes['tag_meat'] + recipes['tag_fish'] + recipes['tag_seafood']\
        + recipes['tag_salmon'] + recipes['tag_beef'] + recipes['tag_steak'] + recipes['tag_bbq']
    
    recipes_new.iloc[:,1:] = recipes_new.iloc[:,1:].astype('bool').astype('float')
    ## filter for image without tags
    tags_new = list(recipes_new.columns)[1:]
    recipes_notag = recipes_new[recipes_new.loc[:,tags_new].sum(1)==0]
    recipes_new = recipes_new[recipes_new.loc[:,tags_new].sum(1)!=0]
    return recipes_new, recipes_notag.loc[:,['photo_id']]
    
# measure the similarity of two sentences
def Similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = torchvision.utils.make_grid(inp)
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    
def train_model(model, dataloaders, criterion, weights ,optimizer, device, num_epochs=25, is_inception=False):
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        since = time.perf_counter()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                        loss = (loss * weights).mean()
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss = (loss * weights).sum()
                    # _, preds = torch.max(outputs, 0)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                # running_corrects += torch.sum(preds == labels.data)
                num_labels = torch.sum(labels.data,1).cpu().numpy().astype(int)
                for i in range(len(num_labels)):
                    preds_indices = torch.topk(outputs[i], num_labels[i]).indices
                    labels_indices = torch.topk(labels.data[i], num_labels[i]).indices
                    numTrue_onelabel = 0
                    if num_labels[i] != 0:
                        for j in range(num_labels[i]):
                            numTrue_onelabel += (preds_indices[j] in labels_indices)
                        running_corrects += numTrue_onelabel/num_labels[i]
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
        time_elapsed = time.perf_counter() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()
    return model_ft, input_size

def Predict(model_ft, recipes, data_dir, num_labels = 2, num_pic = 4):
    batch_size = num_pic
    data_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    pred_data = MyImageDataset.Dataset(
        annotations_file = recipes,
        img_dir= data_dir,
        transform = data_transforms
    )
    pred_dataloader = DataLoader(pred_data, batch_size=batch_size, shuffle=True)
    inputs, classes = next(iter(pred_dataloader))
    outputs = model_ft(inputs.to(device))
    tags_idx = torch.topk(outputs, num_labels).indices.cpu().numpy()
    tags = tag_pred[tags_idx]
    title = ''
    for i in range(num_pic):
        title += ' ' + str(i) + '.'
        for j in range(num_labels):
            title += tags[i][j] + ','
    imshow(inputs,title)
    
if __name__ == '__main__':
    #### Parameters #######
    data_dir = "./recipes"
    # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
    model_name = "resnet"    
    num_classes = 7
    batch_size = 64 
    num_epochs = 50
    feature_extract = True
    test_proportion = 0.2
    seed = 0
    #######################
    
    
    recipes_new, recipes_notag = Preprocesse()
    tag_pred = np.array(recipes_new.columns[1:])
    recipes_train,recipes_val = train_test_split(recipes_new, test_size=test_proportion,random_state=seed)
    
    # Initialize the model for this run
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    
    # Print the model we just instantiated
    print(model_ft)
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        }
    train_data = MyImageDataset.Dataset(
        annotations_file = recipes_train,
        img_dir= data_dir,
        transform = data_transforms['train']
    )
    val_data = MyImageDataset.Dataset(
        annotations_file = recipes_val,
        img_dir= data_dir,
        transform = data_transforms['val']
    )
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    dataloaders_dict = {'train':train_dataloader,'val':val_dataloader}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # training_data.__getitem__(0)
    # inputs, classes = next(iter(train_dataloader))
    # out = torchvision.utils.make_grid(inputs)
    # imshow(out)
    # image = Image.open('recipes\\CEteOEzpt5e.png').convert('RGB')

    model_ft = model_ft.to(device)
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)
    
    # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    optimizer_ft = optim.Adam(params_to_update, lr=0.001)
    
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    # Train and evaluate
    weights = torch.ones(num_classes, dtype=torch.float64, device=device)
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, weights, optimizer_ft, device, 
                                  num_epochs=num_epochs, is_inception=(model_name=="inception"))
    Predict(model_ft, recipes_val, data_dir, num_labels = 1, num_pic = 4)
