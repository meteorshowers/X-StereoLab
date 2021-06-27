import os
import torch
import torch.utils.data as data
import torch
#import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
from . import preprocess
from . import listflowfile as lt
from . import readpfm as rp
import numpy as np
import cv2
import torch.nn.functional as F

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def disparity_loader(path):
    return rp.readPFM(path)

def random_replace(img,num,size):
    #random crop areas and replace to the same size random crop from the image self.
    #from HITNet ,it random crop the right image.
    #args num:num of areas to crop and replace
    #     size: random [0,size]*[0,size]
    h = img.shape[0]
    w = img.shape[1]
    for i in range(num):
        size_ix = random.randint(0,size)
        size_iy = random.randint(0, size)
        x1 = random.randint(0, w - size_ix)
        y1 = random.randint(0, h - size_iy)

        x2 = random.randint(0, w - size_ix)
        y2 = random.randint(0, h - size_iy)
        #replace
        img[y1:y1 + size_iy, x1:x1 + size_ix, :] = img[y2:y2 + size_iy, x2:x2 + size_ix, :]

    return img

class myImageFloder(data.Dataset):
    def __init__(self, left, right, left_disparity,right_disparity, training, loader=default_loader, dploader=disparity_loader):

        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training
        if right_disparity is not None:
            self.disp_R = right_disparity
        else:
            self.disp_R = None

        print('len', len(self.left))
    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        disp_L = self.disp_L[index]
        if self.disp_R is not None:
            disp_R = self.disp_R[index]
            dataR,scaleR = self.dploader(disp_R)
            dataR = np.ascontiguousarray(dataR, dtype=np.float32)

        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL, scaleL = self.dploader(disp_L)
        dataL = np.ascontiguousarray(dataL, dtype=np.float32)

        if self.training:

            h = left_img.shape[0]
            w = left_img.shape[1]

            th, tw = 320, 960

            x1 = random.randint(0, w - tw)
            y1 =\
                random.randint(0, h - th)

            left_img = left_img[y1:y1+th,x1:x1+tw,:]
            right_img = right_img[y1:y1+th,x1:x1+tw,:]

            #left_img = random_replace(left_img,5,80)

            dataL = dataL[y1:y1 + th, x1:x1 + tw]

            if dataR is not None:
                dataR=dataR[y1:y1 + th, x1:x1 + tw]


            processed = preprocess.get_transform(augment=False)

            #random replace
            #right_img = random_replace(right_img,4,5)

            left_img_and_d= processed(image=left_img,mask=dataL,bboxes=[],category_id=[])
            left_img = left_img_and_d['image']
            dataL = left_img_and_d['mask']
            if dataR is not None:
                right_img_and_d = processed(image=right_img, mask=dataR, bboxes=[], category_id=[])
                right_img = right_img_and_d['image']
                dataR = right_img_and_d['mask']
            else:
                right_img = processed(image=right_img,mask=None,bboxes=[],category_id=[])['image']


            if dataR is not None:
                return left_img, right_img, dataL,dataR
            else:
                return left_img, right_img, dataL
        else:

            h = left_img.shape[0]
            w = left_img.shape[1]

            th, tw = 512, 960

            #x1 = random.randint(0, w - tw)
            #y1 =  random.randint(0, h - th)
            x1 = 0
            y1 = 0

            left_img = left_img[y1:y1 + th, x1:x1 + tw, :]
            right_img = right_img[y1:y1 + th, x1:x1 + tw, :]

            dataL = dataL[y1:y1 + th, x1:x1 + tw]
            if dataR is not None:
                dataR=dataR[y1:y1 + th, x1:x1 + tw]

            processed = preprocess.get_transform(augment=False)
            left_img_and_d= processed(image=left_img,mask=dataL,bboxes=[],category_id=[])
            left_img = left_img_and_d['image']
            dataL = left_img_and_d['mask']
            if dataR is not None:
                right_img_and_d = processed(image=right_img, mask=dataR, bboxes=[], category_id=[])
                right_img = right_img_and_d['image']
                dataR = right_img_and_d['mask']
            else:
                right_img = processed(image=right_img,mask=None,bboxes=[],category_id=[])['image']


            if dataR is not None:
                return left_img, right_img, dataL, dataR
            else:
                return left_img, right_img, dataL


    def __len__(self):
        return len(self.left)

