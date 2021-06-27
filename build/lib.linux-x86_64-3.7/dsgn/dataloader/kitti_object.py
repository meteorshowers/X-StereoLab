''' Helper class and functions for loading KITTI objects

Author: Charles R. Qi
Date: September 2017
'''
from __future__ import print_function

import os
import sys
import numpy as np
import cv2
from PIL import Image
import dsgn.dataloader.kitti_util as utils
import scipy.misc as ssc
import imageio

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

class kitti_object(object):
    '''Load and parse object data into a usable format.'''
    
    def __init__(self, split_txt, split_dir, type_whitelist, name_regex='%06d', res_dir=None):
        self.split_dir = split_dir
        self.split = split_txt
        self.type_whitelist = type_whitelist

        self.idxs = [int(line.rstrip().split('.')[0]) for line in open(self.split)]
        self.num_samples = len(self.idxs)
        print('split_txt in {} has {} samples'.format(split_txt, self.num_samples))

        self.image_dir = os.path.join(self.split_dir, 'image_2')
        self.right_image_dir = os.path.join(self.split_dir, 'image_3')
        self.calib_dir = os.path.join(self.split_dir, 'calib')
        self.lidar_dir = os.path.join(self.split_dir, 'velodyne')
        self.lidar_rgb_dir = os.path.join(self.split_dir, 'velodyne_rgb_insimg')
        self.lidar_align_dir = os.path.join(self.split_dir, 'velodyne_align')
        self.label_dir = os.path.join(self.split_dir, 'label_2')
        self.disparity_dir = os.path.join(self.split_dir, 'disparity')
        self.predicted_disparity_dir = os.path.join(self.split_dir, 'predict_disparity')
        self.res_dir = res_dir

        self.name_regex = name_regex

    def __len__(self):
        return self.num_samples

    def get_data_idxs(self):
        return self.idxs

    def get_image_path(self, idx):
        return os.path.join(self.image_dir,  (self.name_regex+'.png')%(idx))

    def get_image(self, idx):
        img_filename = os.path.join(self.image_dir, (self.name_regex+'.png')%(idx))
        return utils.load_image(img_filename)

    def get_right_image(self, idx):
        img_filename = os.path.join(self.right_image_dir, (self.name_regex+'.png')%(idx))
        return utils.load_image(img_filename)
        
    def get_right_image_path(self, idx):
        return os.path.join(self.right_image_dir, (self.name_regex+'.png')%(idx))

    def get_image_shape(self, idx):
        img_filename = os.path.join(self.image_dir, (self.name_regex+'.png')%(idx))
        return Image.open(img_filename).size[::-1] # lazy operation (width, height) -> (height, width)

    def get_right_image_shape(self, idx):
        img_filename = os.path.join(self.right_image_dir, (self.name_regex+'.png')%(idx))
        return Image.open(img_filename).size[::-1] # lazy operation (width, height) -> (height, width)

    # get disparity of left image # generated from LiDAR # sparse
    def get_disparity(self, idx):
        disp_map = np.load(self.disparity_dir + '/' + '{:06d}.npy'.format(idx)).astype(np.float32)
        return disp_map

    # get predicted disparity of left image # dense
    def get_pred_disparity(self, idx):
        pred_disp_map = imageio.imread(self.predicted_disparity_dir + '/' + '{:06d}.png'.format(idx)) / 256.
        return pred_disp_map

    def get_lidar(self, idx): 
        lidar_filename = os.path.join(self.lidar_dir, (self.name_regex+'.bin')%(idx))
        return utils.load_velo_scan(lidar_filename)

    def get_lidar_rgb(self, idx):
        return np.load(os.path.join(self.lidar_rgb_dir, (self.name_regex+'.bin.npy')%(idx)))

    def get_lidar_align_online(self, idx, return_calib=False, extend_bound=400., rgb=False, lidar_sep=1):
        if rgb:
            lidar = self.get_lidar_rgb(idx)
        else:
            lidar = self.get_lidar(idx)
            if lidar_sep > 1:
                num_lidars = 64
                sparse_lidar = []
                for i in range(0, num_lidars, lidar_sep):
                    sparse_lidar.append(lidar[i * len(lidar) // num_lidars : (i+1) * len(lidar) // num_lidars])
                sparse_lidar = np.concatenate(sparse_lidar, axis=0)
                lidar = sparse_lidar

        calib = self.get_calibration(idx)
        lidar_rect = calib.project_velo_to_rect(lidar[:, :3])
        lidar_img = calib.project_rect_to_image(lidar_rect)
        height, width = self.get_image_shape(idx)
        front_img_idx = (lidar_rect[:, 2] >= 0) & (lidar_img[:, 0] >= 0 - extend_bound) & (lidar_img[:, 0] < width + extend_bound) \
            & (lidar_img[:, 1] >= 0 - extend_bound / 3.) & (lidar_img[:, 1] < height + extend_bound / 3.)
        lidar = lidar[front_img_idx]

        if return_calib:
            return lidar, calib
        else:
            return lidar

    def get_calibration(self, idx):
        calib_filename = os.path.join(self.calib_dir, (self.name_regex+'.txt')%(idx))
        return utils.Calibration.fromfile(calib_filename)

    def get_right_calibration(self, idx):
        calib_filename = os.path.join(self.calib_dir, (self.name_regex+'.txt')%(idx))
        return utils.Calibration.fromrightfile(calib_filename)

    def get_label_objects(self, idx):
        assert ('train' in self.split or 'val' in self.split)
        label_filename = os.path.join(self.label_dir, (self.name_regex+'.txt')%(idx))
        return utils.read_label(label_filename)

    def get_result_objects(self, idx):
        label_filename = os.path.join(self.res_dir, (self.name_regex+'.txt')%(idx))
        return utils.read_label(label_filename)
        
    def get_depth_map(self, idx):
        pass

    def get_top_down(self, idx):
        pass
