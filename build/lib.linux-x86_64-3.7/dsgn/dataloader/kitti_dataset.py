import os
import os.path as osp
from dsgn.dataloader.kitti_object import *
from dsgn.dataloader.kitti_util import *
import pickle
from itertools import cycle
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class kitti_dataset(object):
    def __init__(self, split='train', data_path=osp.join(CURRENT_DIR, '..', '..', 'data', 'kitti'), 
            type_whitelist=['Car'],
            min_reflectance=-0.1,
            res_path = None):

        self.data_path = data_path
        self.split = split
        self.type_whitelist = type_whitelist
        self.min_reflectance = min_reflectance
        self.name_regex = '%06d'
        self.res_path = res_path

        if split == 'train':
            self.train_txt = './train.txt'
            self.train_folder = './training'
            self.val_txt = './val.txt'
            self.val_folder = './training'
        elif split == 'test':
            self.train_txt = None
            self.train_folder = None
            self.val_txt = './test.txt'
            self.val_folder = './testing'
        elif split == 'trainval':
            self.train_txt = './trainval.txt'
            self.train_folder = './training'
            self.val_txt = './test.txt'
            self.val_folder = './testing'

        self.train_dataset = None
        self.val_dataset = None

        if self.train_txt:
            self.train_dataset = kitti_object(osp.join(self.data_path, self.train_txt), osp.join(self.data_path, self.train_folder), self.type_whitelist, name_regex=self.name_regex, res_dir=self.res_path)
        if self.val_txt:
            self.val_dataset = kitti_object(osp.join(self.data_path, self.val_txt), osp.join(self.data_path, self.val_folder), self.type_whitelist, name_regex=self.name_regex, res_dir=self.res_path)

    def prep_lidar_data(self, dataset):
        data = dict()

        for data_idx in dataset.get_data_idxs():
            print('processing {:06d}'.format(data_idx))
            calib = dataset.get_calibration(data_idx) # 3  by 4 matrix
            objects = dataset.get_label_objects(data_idx)
            pc_velo = dataset.get_lidar(data_idx)
            pc_rect = np.zeros_like(pc_velo)
            pc_rect[:,0:3] = calib.project_velo_to_rect(pc_velo[:,0:3])
            pc_rect[:,3] = pc_velo[:,3]
            img = dataset.get_image(data_idx)
            img_height, img_width, img_channel = img.shape
            _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(pc_velo[:,0:3],
                calib, 0, 0, img_width, img_height, True, clip_distance=2)

            inds = (pc_velo[:, 3] > self.min_reflectance) & (img_fov_inds)

            points = []
            for p3d, p2d_incam in zip(pc_velo[inds], pc_image_coord[inds]):
                x, y = int(np.round(p2d_incam[0])), int(np.round(p2d_incam[1]))
                if x >= 0 and y >= 0 and x < img_width and y < img_height:
                    point = [*p3d, *img[y, x], *p2d_incam]
                    points.append(point)
            points = np.array(points)

            data[data_idx] = points

        return data
