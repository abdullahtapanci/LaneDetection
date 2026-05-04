import torch
from torch.utils.data import Dataset
import os

#Here I defined my custom dataset. Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable 
#around the Dataset to enable easy access to the samples.
class LaneDataset(Dataset):
    #There must be three functions in dataset class : __init__, __len__ and __getitem__
    #The __init__ function is run once when instantiating the Dataset object.
    def __init__(self, manifest_path, root_dir, transform=None):
        #In manifest_path I actually give the adress of train.txt. This file contains relative paths to the images and their
        #corresponding binary and instance masks. Sample line from train.txt :
        #train_set/clips/0313-1/11100/20.jpg train_set/seg_label/clips/0313-1/11100/20.png train_set/instance_label/clips/0313-1/11100/20.png
        with open(manifest_path, 'r') as f:
            self.instances = f.readlines()
        #root_dir is the path to the tusimple folder. I will use os.path.join to get the full path to each image and mask.
        self.root_dir = root_dir
        #We generally have transform that is applied to the input images, and target_transform that is applied to the labels. In this case, 
        #I will just use transform for both since I will apply the same transformations to images and masks.
        self.transform = transform

    #The __len__ function returns the number of samples in our dataset.
    def __len__(self):
        return len(self.instances)

    #The __getitem__ function loads and returns a sample from the dataset at the given index idx. 
    def __getitem__(self, idx):
        line = self.instances[idx].strip().split()
        img_path = os.path.join(self.root_dir, line[0])
        bin_path = os.path.join(self.root_dir, line[1])
        inst_path = os.path.join(self.root_dir, line[2])

        #Apply transformations
        img, bin_mask, inst_mask = self.transform(img_path, bin_path, inst_path)

        return img, bin_mask, inst_mask
    
    #The Dataset retrieves our dataset’s features and labels one sample at a time. While training a model, we typically want to 
    #pass samples in “minibatches”, reshuffle the data at every epoch to reduce model overfitting, and use Python’s multiprocessing 
    #to speed up data retrieval.
    #DataLoader is an iterable that abstracts this complexity for us in an easy API.

