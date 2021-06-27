
import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import numpy as np
#from dataloader.preprocess import preprocess
import dataloader.preprocess as preprocess

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert('RGB')


def disparity_loader(path):
    return Image.open(path)



def dataloader(filepath):

  left_fold  = 'left/'
  right_fold = 'right/'


  left_test= [img for img in os.listdir(filepath+left_fold) if img.find('_left') > -1]
  left_test.sort()
  right_test= [img for img in os.listdir(filepath+right_fold) if img.find('_right') > -1]
  right_test.sort()

  left_test = [filepath+left_fold+img for img in left_test]
  right_test = [filepath+right_fold+img for img in right_test]

  return left_test, right_test

class myImageFloder(data.Dataset):
    def __init__(self, left, right, loader=default_loader):

        self.left = left
        self.right = right
        self.loader = loader


    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        print('left',index,left)
        print('right',index,right)


        left_img = self.loader(left)
        right_img = self.loader(right)


        #test   not for training
        w, h = left_img.size

        left_img = left_img.crop((w - 992, h - 736, w, h))
        right_img = right_img.crop((w - 992, h - 736, w, h))
        # left_img = left_img.crop((w - 1232, h - 368, w, h))
        # right_img = right_img.crop((w - 1232, h - 368, w, h))
        w1, h1 = left_img.size

        #dataL = dataL.crop((w - 1232, h - 368, w, h))


        processedL = preprocess.get_transform(augment=False,camera=None)
        processedR = preprocess.get_transform(augment=False,camera=None)
        left_img = processedL(left_img)
        right_img = processedR(right_img)

        return left_img, right_img

    def __len__(self):
        return len(self.left)

if __name__ == '__main__':
    left,right=dataloader('/disk1/hyj/test_picture/819_testpic/')
    print(left)
    print(len(left))
    print(right)
    print(len(right))