
# coding: utf-8

# # 2D Unet for Biomedical Image Segmentation

# ## Python Imports

# In[1]:


import numpy as np
import os
import sys
from matplotlib import pyplot as plt


import torch
from torch import nn
from torch.autograd import Variable
from torchvision import transforms, utils
from skimage import io, transform
from skimage.color import rgb2gray

from models import * 
from train import *
from data_loader import * 

from scipy import ndimage

from matplotlib.colors import ListedColormap



# ##  Data Loading and Augmentation

# In[5]:


img_resize = 512
batch_size = 2

# traning set
data_dir = './datasets/mitochondria/training/images/'
label_dir = './datasets/mitochondria/training/labels/'
MitoDataset = EM_Dataset(data_dir, label_dir, img_resize)
MitoDataLoader = DataLoader(MitoDataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=4)

# validation set 
val_data_dir = './datasets/mitochondria/validation/images/'
val_label_dir = './datasets/mitochondria/validation/labels/'
MitoValDataset = EM_Dataset(val_data_dir, val_label_dir, img_resize, do_transform=False)
MitoValDataLoader = DataLoader(MitoValDataset,
                             batch_size=1,
                             shuffle=True,
                             num_workers=4)


for i, data in enumerate(MitoDataLoader):

        print("training image from batch %d" % i)

        imgs = data[0]
        labels = data[1]

       
        mask, nlabels = ndimage.label(labels[0])

        
        rand_cmap = ListedColormap(np.random.rand(256,3))

        labels_for_display = np.where(mask > 0, mask, np.nan)

        f, (ax1,ax2,ax3) = plt.subplots(1,3, sharey=True,figsize=(15,5), dpi=80)

        ax1.imshow(np.squeeze(imgs[0]), cmap="gray")
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        ax2.imshow(np.squeeze(labels[0]), cmap="gray")
        ax2.set_title("Ground Truth Mask")
        ax2.axis('off')
        
        ax3.imshow(np.squeeze(imgs[0]), cmap='gray')
        ax3.imshow(np.squeeze(labels_for_display), cmap=rand_cmap)
        ax3.set_title("Instance Segmentation (%d nuclei)"% nlabels)
        ax3.axis('off')
        

        plt.show()
       


# In[4]:


# Prep GPU
GPU = torch.cuda.is_available()
print("GPU is {}enabled ".format(['not ', ''][GPU]))


n_epochs = 500

# unet(n_channels, n_classes, n_filters_start=64 )
un = unet(1,1)
if GPU: 
    un = un.cuda()

optimizer = torch.optim.Adam(un.parameters(), lr=0.01)
criterion = nn.BCELoss()

output_dir = "./2D_Unet_output/"


# run training 
training(GPU, MitoDataLoader, MitoValDataLoader, MitoValDataset, un, optimizer, criterion, n_epochs, batch_size, output_dir, warm_start=False)

