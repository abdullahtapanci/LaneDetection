import cv2
import DeepLearningTechnique.src.config as cfg
import numpy as np
import torch
import random

def transform_image(image_path, bin_path, inst_path, training=True):
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
    if training:
        img = color_jitter(img, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        img = random_blur(img)
        img, bin_mask, inst_mask = random_horizontal_flip(img, bin_mask, inst_mask, p=0.5)
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


def seeTransforms(image_path, bin_path, inst_path):
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

    #This part contains the data augmentation steps. Mirror the training pipeline used in transform_image.
    img = color_jitter(img, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
    img = random_blur(img)
    img, bin_mask, inst_mask = random_horizontal_flip(img, bin_mask, inst_mask, p=0.5)
    img, bin_mask, inst_mask = random_translate(img, bin_mask, inst_mask)
    img, bin_mask, inst_mask = random_perspective(img, bin_mask, inst_mask)

    return img, bin_mask, inst_mask




#Data Augmentation functions
def color_jitter(img, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1):
    """
    Mirrors torchvision.transforms.ColorJitter for a uint8 RGB numpy image.
    Each parameter f sets a sampling range:
        brightness factor ~ U(1-f, 1+f)
        contrast   factor ~ U(1-f, 1+f)
        saturation factor ~ U(1-f, 1+f)
        hue        shift  ~ U(-h, h)  in fraction of full hue circle

    The four ops are applied in a random order each call, like torchvision.
    Input/output are uint8 RGB so the rest of the pipeline (normalization)
    stays unchanged.
    """
    img = img.astype(np.float32)
    ops = ['brightness', 'contrast', 'saturation', 'hue']
    random.shuffle(ops)

    for op in ops:
        if op == 'brightness' and brightness > 0:
            factor = random.uniform(max(0.0, 1 - brightness), 1 + brightness)
            img = np.clip(img * factor, 0, 255)

        elif op == 'contrast' and contrast > 0:
            factor = random.uniform(max(0.0, 1 - contrast), 1 + contrast)
            #Contrast is rescaling around the per-channel mean — same convention as torchvision.
            mean = img.mean(axis=(0, 1), keepdims=True)
            img = np.clip((img - mean) * factor + mean, 0, 255)

        elif op == 'saturation' and saturation > 0:
            factor = random.uniform(max(0.0, 1 - saturation), 1 + saturation)
            #Blend with the grayscale version of the image. factor=0 → pure grayscale, factor=1 → original.
            gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
            gray = gray[..., None]   #(H, W, 1) so it broadcasts across the 3 channels
            img = np.clip(gray + (img - gray) * factor, 0, 255)

        elif op == 'hue' and hue > 0:
            #OpenCV stores H in [0, 180) for 8-bit. Shift cyclically.
            shift = int(round(random.uniform(-hue, hue) * 180))
            hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV)
            hsv[..., 0] = (hsv[..., 0].astype(np.int16) + shift) % 180
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32)

    return img.astype(np.uint8)


def random_horizontal_flip(img, bin_mask, inst_mask, p=0.5):
    """
    Flip image and both masks together so they stay aligned. Lane instance IDs
    don't carry semantic meaning (the discriminative loss only groups by unique
    ID), so flipping a left-most lane to right-most is fine.
    """
    if random.random() < p:
        img       = cv2.flip(img, 1)
        bin_mask  = cv2.flip(bin_mask, 1)
        inst_mask = cv2.flip(inst_mask, 1)
    return img, bin_mask, inst_mask


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

        delta = 10

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


