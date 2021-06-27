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

def dataloader(filepath):

  left_fold  = 'image_2/'
  right_fold = 'image_3/'
  disp_L = 'disp_occ_0/'
  disp_R = 'disp_occ_1/'
  disp_norm = 'dispnorm_occ/'

  image = [img for img in os.listdir(filepath+left_fold) if img.find('_10') > -1]

  all_index = np.arange(200)
  #np.random.shuffle(all_index)
  # vallist = all_index[:40]
#   val = ['{:06d}_10.png'.format(x) for x in vallist]
  

  val = []
  train = [x for x in image if x not in val]
  random.shuffle(train)


  left_train  = [filepath+left_fold+img for img in train]
  right_train = [filepath+right_fold+img for img in train]
  disp_train_L = [filepath+disp_L+img for img in train]
  disp_train_R = [filepath+disp_R+img for img in train]
  norm_train_L = [filepath+disp_norm+img for img in train]

  left_val  = [filepath+left_fold+img for img in val]
  right_val = [filepath+right_fold+img for img in val]
  disp_val_L = [filepath+disp_L+img for img in val]
  disp_val_R = [filepath+disp_R+img for img in val]
  norm_val_L = [filepath+disp_norm+img for img in val]

  return left_train, right_train, disp_train_L, norm_train_L, left_val, right_val, disp_val_L, norm_val_L
