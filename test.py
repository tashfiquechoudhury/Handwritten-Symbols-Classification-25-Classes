# -*- coding: utf-8 -*-
"""
File: test.py
Desc: Testing code for the symbols classification problem
Author: Tashfique Hasnine Choudhury
"""
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
import urllib.request
torch.manual_seed(17)

#Downloading the .pth file
model_url = "https://dl.dropboxusercontent.com/s/4jm7u6tw9gdvkh3/resnet18_50-Ag2_epochs_saved_weights.pth?dl=0"
print('Downloading file...')
urllib.request.urlretrieve(model_url, 'C:\\Users\\HP\\Documents\\MLProject\\resnet18_50-Ag2_epochs_saved_weights.pth') #Set directory
print('Download completed')

def Test_func(test_images):
    class CustomTensorDataset(Dataset):
    
        def __init__(self, data, labels=None, transform=None):      
            self.data = data
            self.labels = labels
            self.transform = transform
    
        def __getitem__(self, index):       
            x = self.data[index]
            
            if self.transform is not None:
                x = self.transform(x)
            if self.labels is not None:
                y = self.labels[index]
                return x, y
            else:
                return x
    
        def __len__(self):    
            return self.data.size(0)
    
    class Block(nn.Module):
        
        def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
            super(Block, self).__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU()
            self.identity_downsample = identity_downsample
            
        def forward(self, x):
            identity = x
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            if self.identity_downsample is not None:
                identity = self.identity_downsample(identity)
            x += identity
            x = self.relu(x)
            return x
    
    class ResNet_18(nn.Module):
        
        def __init__(self, image_channels, num_classes):
            
            super(ResNet_18, self).__init__()
            self.in_channels = 64
            self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU()
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            
            #resnet layers
            self.layer1 = self.__make_layer(64, 64, stride=1)
            self.layer2 = self.__make_layer(64, 128, stride=2)
            self.layer3 = self.__make_layer(128, 256, stride=2)
            self.layer4 = self.__make_layer(256, 512, stride=2)
            
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, num_classes)
            
        def __make_layer(self, in_channels, out_channels, stride):
            
            identity_downsample = None
            if stride != 1:
                identity_downsample = self.identity_downsample(in_channels, out_channels)
                
            return nn.Sequential(
                Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride), 
                Block(out_channels, out_channels)
            )
            
        def forward(self, x):
            
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            
            x = self.avgpool(x)
            x = x.view(x.shape[0], -1)
            x = self.fc(x)
            return x 
        
        def identity_downsample(self, in_channels, out_channels):
            
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1), 
                nn.BatchNorm2d(out_channels)
            )
    
    model = ResNet_18(1, 25)
    
    
    test_images = (np.load(f'{test_images}.npy')).astype('float32')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load('resnet18_50-Ag2_epochs_saved_weights.pth', map_location=device))
    
    def crop_center(img,cropx,cropy):
        img=cv2.resize(img,(cropx,cropy))
        return img
    
    img=[]
    for i in range(0,len(test_images)):
        img.append(crop_center(test_images[i],140,140))
    test_images=np.array(img)
    
    
    test = torch.tensor(test_images)/255.0
    test = torch.from_numpy(np.asarray(test)).view(-1, 1, 140, 140)
    
    #prepare a dataloader with the test set
    testset = CustomTensorDataset(test, None)
    test_loader = DataLoader(testset, batch_size=32, shuffle=False)
    device = torch.device("cpu")
    model.eval()
    labels = []
    for inputs in test_loader:
        inputs = transforms.functional.resize(inputs, (112, 112))
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)
        predictions = predictions.to("cpu")
        labels.extend(predictions.numpy())
        
    return np.array(labels)

labels = Test_func('Images_Sample_Test') 
print(labels)


def evaluation(test_labels):
    
    test_labels = np.load(f'{test_labels}.npy')
    print('Accuracy:',accuracy_score(test_labels, labels))
    print(classification_report(test_labels, labels))
    df_cfm = pd.DataFrame(confusion_matrix(test_labels, labels), index = np.array(range(0,25)), columns = np.array(range(0,25))).astype('int')
    plt.figure(figsize = (15,10))
    heatmap = sns.heatmap(df_cfm, annot=True)
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=15)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right',fontsize=15)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
#    plt.savefig('Test')
    return

evaluation('Labels_Sample_Test')