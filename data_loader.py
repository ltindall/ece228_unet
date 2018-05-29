import os
from skimage import io
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import sys
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from PIL import Image
import random
import torch





class EM_Dataset(Dataset):
    def __init__(self, data_dir, label_dir, image_resize, do_transform=True):
        
        self.image_resize = image_resize 
        self.do_transform = do_transform
        
        self.data_dir = data_dir
        self.data_file_list = [f for f in os.listdir(self.data_dir) if os.path.isfile(os.path.join(self.data_dir, f))]
        self.data_file_list.sort()
        
        self.label_dir = label_dir
        self.label_file_list = [f for f in os.listdir(self.label_dir) if os.path.isfile(os.path.join(self.label_dir, f))]
        self.label_file_list.sort()
        
        assert (len(self.data_file_list) == len(self.label_file_list)),"Image and label count must match!"
        
    def __len__(self):
        return int(len(self.data_file_list))

    def __getitem__(self, idx):

        data_filename = self.data_file_list[idx]
        img = io.imread(self.data_dir + data_filename)
    
        label_filename = self.label_file_list[idx]
        label = rgb2gray(io.imread(self.label_dir + label_filename))
        #print("max label1 = ",np.max(label))
        #print("min label1 = ",np.min(label))
        label[label!=0]=255
        
        
        
        img = Image.fromarray(img.astype(np.uint8))
        label = Image.fromarray(label.astype(np.uint8))
        
        
        
        #print("max label = ",np.max(label))
        #print("min label = ",np.min(label))

        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        toTensor = transforms.ToTensor()
        
        
        if self.do_transform:
            #cj = transforms.ColorJitter(brightness=1.0, contrast=1.0, saturation=1.0, hue=0.2)
            
            horizontal_flip = transforms.RandomHorizontalFlip()
            rotate = transforms.RandomRotation(90)
            
            resize = transforms.RandomResizedCrop(self.image_resize)
            #resize = transforms.Resize(self.image_resize)

            self.transform = [horizontal_flip, resize, toTensor]
        else:

            resize = transforms.Resize(self.image_resize)
            self.transform = [resize, toTensor]

        for i,t in enumerate(self.transform): 

            seed = random.randint(0,2**32)

            random.seed(seed)
            img = t(img)

            
            random.seed(seed)
            label = t(label)

        img = normalize(img)
        
        return img,label

