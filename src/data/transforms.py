import cv2
import src.config as cfg
import numpy as np
import torch
import random

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

    #This part contains the data augmentation steps.
    img = random_brightness(img)
    img = random_blur(img)
    img, bin_mask, inst_mask = random_translate(img, bin_mask, inst_mask)
    img, bin_mask, inst_mask = random_perspective(img, bin_mask, inst_mask)

    #Here we normalize the image
    img = img.astype(np.float32) / 255.0
    #We apply ImageNet normalization. This is a common practice when using pretrained models, as it helps the model to 
    #generalize better by normalizing the input data to have a similar distribution as the data it was originally 
    #trained on (ImageNet). The mean and std values are calculated from the ImageNet dataset and are used to normalize 
    #each channel of the image.
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    #Here we perform standardization (also caled z-score normalization) by subtracting the mean and dividing by the 
    #standard deviation for each channel of the image. img- mean gives us the centered data (mean of 0), and dividing by 
    #std scales the data fall between -1 and 1 (for most pixel values), which can help the model to converge faster 
    #during training.
    img = (img - mean) / std

    #We apply ToTensor step here. PyTorch models expect input in the form of (C, H, W) and masks should have a channel dimension as well.
    #For the image, we permute the dimensions from (H, W, C) to (C, H, W). For the masks, we add a channel dimension using unsqueeze(0).
    img = torch.from_numpy(img).permute(2, 0, 1)
    bin_mask = torch.from_numpy(bin_mask).unsqueeze(0)
    inst_mask = torch.from_numpy(inst_mask).unsqueeze(0)

    return img, bin_mask, inst_mask




#Data Augmentation functions
def random_brightness(img):

    if random.random() < 0.5:

        factor = random.uniform(0.7, 1.3)

        img = img.astype(np.float32) * factor
        img = np.clip(img, 0, 255)

    return img.astype(np.uint8)

def random_blur(img):

    if random.random() < 0.3:
        img = cv2.GaussianBlur(img, (5,5), 0)

    return img

def random_translate(img, bin_mask, inst_mask):

    if random.random() < 0.5:

        tx = random.randint(-20, 20)
        ty = random.randint(-10, 10)

        M = np.float32([
            [1, 0, tx],
            [0, 1, ty]
        ])

        h, w = img.shape[:2]

        img = cv2.warpAffine(
            img,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )

        bin_mask = cv2.warpAffine(
            bin_mask,
            M,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT
        )

        inst_mask = cv2.warpAffine(
            inst_mask,
            M,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT
        )

    return img, bin_mask, inst_mask


def random_perspective(img, bin_mask, inst_mask):

    if random.random() < 0.5:

        h, w = img.shape[:2]

        src = np.float32([
            [0,0],
            [w,0],
            [0,h],
            [w,h]
        ])

        delta = 40

        dst = np.float32([
            [random.randint(0, delta), random.randint(0, delta)],
            [w-random.randint(0, delta), random.randint(0, delta)],
            [random.randint(0, delta), h-random.randint(0, delta)],
            [w-random.randint(0, delta), h-random.randint(0, delta)]
        ])

        M = cv2.getPerspectiveTransform(src, dst)

        img = cv2.warpPerspective(
            img,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR
        )

        bin_mask = cv2.warpPerspective(
            bin_mask,
            M,
            (w, h),
            flags=cv2.INTER_NEAREST
        )

        inst_mask = cv2.warpPerspective(
            inst_mask,
            M,
            (w, h),
            flags=cv2.INTER_NEAREST
        )

    return img, bin_mask, inst_mask


