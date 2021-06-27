import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import numpy as np
from ..utils import preprocess

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    
    return Image.open(path).convert('RGB')


def npy_loader(path):
    return np.load(path)

def disparity_loader(path):
    return Image.open(path)


class myImageFloder(data.Dataset):
    def __init__(self, left, right, left_disparity, left_norm, training, loader=default_loader, dploader=disparity_loader):

        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.norm_L = left_norm
        self.loader = loader
        self.dploader = dploader
        self.npy_loader = npy_loader
        self.training = training

    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        disp_L = self.disp_L[index]
        norm_L = self.norm_L[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL = self.dploader(disp_L)
        normL = self.npy_loader(norm_L[:-3]+'npy')
        

        if self.training:
            w, h = left_img.size
            # th, tw = 320, 1152
            # th, tw = 256, 1152
            # th, tw = 311, 1178
            th, tw = 320, 1152
            # th, tw = 256, 512
            

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

            dataL = np.ascontiguousarray(dataL, dtype=np.float32) / 256
            dataL = dataL[y1:y1 + th, x1:x1 + tw]

            
            normL = normL[y1:y1 + th, x1:x1 + tw, :]

            processed = preprocess.get_transform(augment=True)
            left_img = processed(left_img)
            right_img = processed(right_img)
            # left_img = left_img/255 - 1
            # right_img = right_img/255 - 1

            # left_img, rigt_img = preprocess.get_transform_unsym(left_img, right_img, [th, tw])
            # left_img, right_img = left_img-1, right_img-1

            # delta_h = np.floor(np.random.uniform(50,150))
            # delta_w = np.floor(np.random.uniform(50,200))

            delta_h = np.floor(np.random.uniform(50,180))
            delta_w = np.floor(np.random.uniform(50,250))
            x1_aug = random.randint(0, th - delta_h)
            y1_aug = random.randint(0, tw - delta_w)
            x2_aug = random.randint(0, th - delta_h)
            y2_aug = random.randint(0, tw - delta_w)
            right_img[:,int(x1_aug):int(x1_aug+delta_h), int(y1_aug):int(y1_aug+delta_w)]  = right_img[:,int(x2_aug):int(x2_aug+delta_h), int(y2_aug):int(y2_aug+delta_w)]

            



            return [left_img.unsqueeze(0), right_img.unsqueeze(0), torch.tensor(dataL).unsqueeze(0),torch.tensor(normL)]
        else:
            w, h = left_img.size
            # left_img = left_img.crop((w - 1232, h - 368, w, h))
            # right_img = right_img.crop((w - 1232, h - 368, w, h))
            # left_img = left_img.crop((w - 1152, h - 256, w, h))
            # right_img = right_img.crop((w - 1152, h - 256, w, h))
            left_img = left_img.crop((w - 1152, h - 320, w, h))
            right_img = right_img.crop((w - 1152, h - 320, w, h))
            w1, h1 = left_img.size

            # dataL = dataL.crop((w - 1152, h - 256, w, h))
            dataL = dataL.crop((w - 1152, h - 320, w, h))
            dataL = np.ascontiguousarray(dataL, dtype=np.float32) / 256

            processed = preprocess.get_transform(augment=False)
            left_img = processed(left_img)
            right_img = processed(right_img)
            # print(left_img, right_img, dataL)

            return [left_img, right_img, dataL, dataL]

    def __len__(self):
        return len(self.left)