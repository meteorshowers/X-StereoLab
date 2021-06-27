import torch
#import torchvision.transforms as transforms
import random

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}

#__imagenet_stats = {'mean': [0.5, 0.5, 0.5],
#                   'std': [0.5, 0.5, 0.5]}

__imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}



def totensor_normalize():

    return A.Compose([
        # A.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225]),
        ToTensorV2(always_apply=True)
    ],p=1)



def augmentv1():
    photometric  = [
        A.Blur(p=0.5),
        A.HueSaturationValue(20,30,20,p=0.5),
        A.RandomBrightnessContrast(0.2,p=0.5),
        A.RandomGamma(p=0.5),
        #A.ISONoise(p=1),
        A.GaussNoise(p=0.5),
        # A.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225],
        # ),
        ToTensorV2()
    ]

    geometric = [
        # A.OpticalDistortion(distort_limit=0.3, shift_limit=0.3,p=1)
        A.ShiftScaleRotate(shift_limit=0.01,scale_limit=0.01,rotate_limit=5,p=0.5)
        #A.ShiftScaleRotate(shift_limit=0.3, scale_limit=0.3, rotate_limit=30, p=0.5)
    ]

    return A.Compose(photometric)



def get_transform(augment=True):


    if augment:
            return augmentv1()
    else:
            return totensor_normalize()






