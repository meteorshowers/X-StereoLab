import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
import random

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath, arg=False):

  left_fold  = 'colored_0/'
  right_fold = 'colored_1/'
  disp_noc   = 'disp_occ/'
  disp_norm   = 'dispnorm_occ/'

  image = [img for img in os.listdir(filepath+left_fold) if img.find('_10') > -1]
  
  valist = [1,15,39,65,101,113,134,154,175,4,16,40,66,102,118,139,156,180,5,19,52,82,104,
            119,143,157,181,9,25,56,85,105,120,145,161,186,11,29,60,89,107,122,148,167,
            188,12,31,63,95,108,128,151,170,14,32,64,97,112,132,153,171]
  # valist = []
  train = []
  val = []
  for i in range(len(image)):
    if i in valist:
      val.append(image[i])
    else:
      train.append(image[i])
  random.shuffle(train)
  
  left_train  = [filepath+left_fold+img for img in train]
  right_train = [filepath+right_fold+img for img in train]
  disp_train = [filepath+disp_noc+img for img in train]
  norm_train = [filepath+disp_norm+img for img in train]


  left_val  = [filepath+left_fold+img for img in val]
  right_val = [filepath+right_fold+img for img in val]
  disp_val = [filepath+disp_noc+img for img in val]
  norm_val = [filepath+disp_norm+img for img in val]

  return left_train, right_train, disp_train, norm_train, left_val, right_val, disp_val, norm_val
