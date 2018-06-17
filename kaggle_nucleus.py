
# coding: utf-8

# # ECE 228 Unet (Kaggle nucleus dataset)

# # Python imports

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



# # Load nucleus data

# In[2]:


# X_train = np.load('datasets/nucleus_train_images.npy')
# X_train = (X_train/127)-1
# Y_train = np.load('datasets/nucleus_train_labels.npy').astype(np.uint8)

# print("Training images shape = ",X_train.shape)
# print("Training labels shape = ",Y_train.shape)
# original_imgs = (127*(np.moveaxis(X_train, 1, -1)+1)).astype(np.uint8)

# i = np.random.randint(len(original_imgs))
# plt.imshow(original_imgs[i])
# plt.title("Train image")
# plt.colorbar()
# plt.show()

# plt.imshow(np.squeeze(Y_train[i,:,:,:]), cmap="gray")
# plt.title("Train label")
# plt.colorbar()
# plt.show()



# In[8]:


img_resize = 128
batch_size = 1


images = np.load('datasets/nucleus_train_images.npy')
masks = np.load('datasets/nucleus_train_labels.npy').astype(np.uint8)

# split into training and validation set 
total_size = images.shape[0]
p = np.random.permutation(total_size)
images = images[p]
masks = masks[p]

val_size = int(0.6*total_size)

X_val = images[:val_size]
Y_val = masks[:val_size]

X_train = images[val_size:]
Y_train = masks[val_size:]



# traning set

KaggleDataset = Kaggle_Dataset(X_train, Y_train, img_resize)
KaggleDataLoader = DataLoader(KaggleDataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=4)

# validation set 
KaggleValDataset = Kaggle_Dataset(X_val, Y_val, img_resize, do_transform=False)
KaggleValDataLoader = DataLoader(KaggleValDataset,
                             batch_size=1,
                             shuffle=True,
                             num_workers=4)



for i, data in enumerate(KaggleDataLoader):

        print("training image from batch %d" % i)
        
    

        imgs = np.array(data[0])
        labels = data[1]

        print(np.max(imgs[0]))
       
        mask, nlabels = ndimage.label(labels[0])

        
        rand_cmap = ListedColormap(np.random.rand(256,3))

        labels_for_display = np.where(mask > 0, mask, np.nan)

        f, (ax1,ax2,ax3) = plt.subplots(1,3, sharey=True,figsize=(15,5), dpi=80)

        ax1.imshow(np.moveaxis(imgs[0],0,-1))
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        ax2.imshow(np.squeeze(labels[0]), cmap="gray")
        ax2.set_title("Ground Truth Mask")
        ax2.axis('off')
        
        ax3.imshow(np.moveaxis(imgs[0],0,-1))
        ax3.imshow(np.squeeze(labels_for_display), cmap=rand_cmap)
        ax3.set_title("Instance Segmentation (%d nuclei)"% nlabels)
        ax3.axis('off')
        

        plt.show()
        
        if i > 1: 
            break
       


# # Run Training

# In[4]:


# Prep GPU
GPU = torch.cuda.is_available()
print("GPU is {}enabled ".format(['not ', ''][GPU]))


n_epochs = 300


#unet(n_channels, n_classes, n_filters_start=64 )
un = unet(3,1)
if GPU: 
    un = un.cuda()

optimizer = torch.optim.Adam(un.parameters(), lr=0.01)
criterion = nn.BCELoss()





# # split into training and validation set 
# total_train_size = X_train.shape[0]
# p = np.random.permutation(total_train_size)
# X_train = X_train[p]
# Y_train = Y_train[p]

# val_size = int(0.2*total_train_size)

# X_val = X_train[:val_size]
# Y_val = Y_train[:val_size]

# X_train = X_train[val_size:]

# Y_train = Y_train[val_size:]

print("80/20 training/validation split")
print("total size = ",total_size)
print("train size = ", total_size - val_size)
print("val size = ", val_size)


for i, data in enumerate(KaggleDataLoader):
    if i == 0: 
        imgs = np.array(data[0])
        print(imgs.shape)
        break
# run training 
#training(GPU, un,X_train, Y_train, X_val, Y_val, optimizer, criterion, n_epochs, batch_size)

output_dir = "./Kaggle_output/"


# run training 
training(GPU, KaggleDataLoader, KaggleValDataLoader, KaggleValDataset, un, optimizer, criterion, n_epochs, batch_size, output_dir, warm_start=False)

