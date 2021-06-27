from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import skimage
import skimage.io
import skimage.transform
import numpy as np
import time
import math
from dsgn.utils import preprocess 
from dsgn.models import *
from dsgn.utils.numpy_utils import *

from env_utils import *

from dsgn.models.inference3d import make_fcos3d_postprocessor

import scipy.misc as ssc
# 2012 data /media/jiaren/ImageNet/data_scene_flow_2012/testing/

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('-cfg', '--cfg', '--config', default='./configs/default/config_car.py', help='config path')
parser.add_argument('--data_path', default='./data/kitti/training',
                    help='select model')
parser.add_argument('--split_file', default='./data/kitti/train.txt',
                    help='split file')
args = parser.parse_args()

exp = Experimenter('./outputs/temp', cfg_path=args.cfg)
cfg = exp.config

from dsgn.dataloader import KITTILoader3D as ls
from dsgn.dataloader import KITTILoader_dataset3d as DA

all_left_img, all_right_img, all_left_disp, = ls.dataloader(args.data_path,
                                                            args.split_file,
                                                            depth_disp=cfg.eval_depth,
                                                            cfg=cfg,
                                                            is_train=False,
                                                            generate_target=True)

class BatchCollator(object):
    def __call__(self, batch):
        transpose_batch = list(zip(*batch))
        l = torch.cat(transpose_batch[0], dim=0)
        r = torch.cat(transpose_batch[1], dim=0)
        disp = torch.stack(transpose_batch[2], dim=0)
        calib = transpose_batch[3]
        calib_R = transpose_batch[4]
        image_sizes = transpose_batch[5]
        image_indexes = transpose_batch[6]
        outputs = [l, r, disp, calib, calib_R, image_sizes, image_indexes]
        return [l, r, disp, calib, calib_R, image_sizes, image_indexes]

ImageFloader = DA.myImageFloder(all_left_img, all_right_img, all_left_disp, False, split=args.split_file, cfg=cfg,
    generate_target=True)

TestImgLoader = torch.utils.data.DataLoader(
    ImageFloader,
    batch_size=1, shuffle=False, num_workers=0, drop_last=False,
    collate_fn=BatchCollator())

cfg.flip_this_image = False

for batch_idx, (imgL, imgR, gt_disp, calib_batch, calib_R_batch, image_sizes, image_indexes) \
    in enumerate(TestImgLoader):

    pass

if cfg.flip:
    
    cfg.flip_this_image = True

    for batch_idx, (imgL, imgR, gt_disp, calib_batch, calib_R_batch, image_sizes, image_indexes) \
        in enumerate(TestImgLoader):

        pass
