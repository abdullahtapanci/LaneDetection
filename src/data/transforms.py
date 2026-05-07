import cv2
import src.config as cfg
import numpy as np
import torch

def transform_image(image_path, bin_path, inst_path):
    #Here we load and resize image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

    #Here we load and resize binary mask
    bin_mask = cv2.imread(bin_path, cv2.IMREAD_GRAYSCALE)
    #We use INTER_NEAREST for masks to avoid introducing new pixel values during resizing.
    bin_mask = cv2.resize(bin_mask, (cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)
    #We binarize the mask (0 for background, 1 for lane) and convert to float32 for PyTorch compatibility.
    bin_mask = (bin_mask > 0).astype(np.float32)

    #Here we load and resize instance mask. Each lane instance has a unique pixel value. We didn't need to binarize this 
    #mask since we want to keep which pixel belongs to which lane.
    inst_mask = cv2.imread(inst_path, cv2.IMREAD_GRAYSCALE)
    inst_mask = cv2.resize(inst_mask, (cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)

    #Here we normalize the image
    img = img.astype(np.float32) / 255.0
    # Apply ImageNet normalization here...
    mean = np.array([0.485, 0.456, 0.406])  # in RGB
    std  = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std

    #We apply ToTensor step here. PyTorch models expect input in the form of (C, H, W) and masks should have a channel dimension as well.
    #For the image, we permute the dimensions from (H, W, C) to (C, H, W). For the masks, we add a channel dimension using unsqueeze(0).
    img = torch.from_numpy(img).permute(2, 0, 1)
    bin_mask = torch.from_numpy(bin_mask).unsqueeze(0)
    inst_mask = torch.from_numpy(inst_mask).unsqueeze(0)

    return img, bin_mask, inst_mask