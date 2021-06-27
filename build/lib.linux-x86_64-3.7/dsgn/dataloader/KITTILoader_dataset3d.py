import random
import time
import os

import numpy as np
from . import preprocess
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import hflip
import torch.utils.data as data
from PIL import Image

from scipy import sparse

from dsgn.utils.numpy_utils import *
from dsgn.utils.numba_utils import *
from dsgn.utils.torch_utils import *

from dsgn.dataloader.kitti_dataset import kitti_dataset
from dsgn.dataloader.KITTILoader3D import get_kitti_annos
from dsgn.utils.bounding_box import Box3DList, compute_corners, quan_to_angle, \
    angle_to_quan, quan_to_rotation, compute_corners_sc

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')

def disparity_loader(path):
    return np.load(path).astype(np.float32)

def convert_to_viewpoint_torch(alpha, z, x):
    return alpha + torch.atan2(z, x) - np.pi / 2

def convert_to_ry_torch(alpha, z, x):
    return alpha - torch.atan2(z, x) + np.pi / 2

class myImageFloder(data.Dataset):
    def __init__(self, left, right, left_disparity, training, loader=default_loader, dploader=disparity_loader, 
            split=None, cfg=None, generate_target=False):
        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training
        self.cfg = cfg
        self.num_classes = self.cfg.num_classes
        self.num_angles = self.cfg.num_angles

        if 'train.txt' in split:
            self.kitti_dataset = kitti_dataset('train').train_dataset
        elif 'val.txt' in split:
            self.kitti_dataset = kitti_dataset('train').val_dataset
        elif 'trainval.txt' in split:
            self.kitti_dataset = kitti_dataset('trainval').train_dataset
        elif 'test.txt' in split:
            self.kitti_dataset = kitti_dataset('trainval').val_dataset

        self.generate_target = generate_target
        self.save_path = './outputs/temp/anchor_{}angles'.format(self.cfg.num_angles)
        self.flip = getattr(self.cfg, 'flip', False)

        if 'trainval.txt' in split:
            self.save_path += '_trainval'
        self.valid_classes = getattr(self.cfg, 'valid_classes', None)
        if self.valid_classes is not None:
            self.save_path += '_validclass_{}'.format('_'.join(list(
                map(lambda x:str(x), self.valid_classes)
            )))
        
        if 2 in self.valid_classes:
            self.less_car = getattr(cfg, 'less_car_pos', False)
            if self.less_car:
                self.save_path += '_lesscar'

        if 1 in self.valid_classes or 3 in self.valid_classes:
            self.less_human = getattr(cfg, 'less_human_pos', False)
            if self.less_human:
                self.save_path += '_lesshuman'

        if self.generate_target:
            os.system('mkdir {}'.format(self.save_path))

    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        disp_L = self.disp_L[index]

        image_index = int(left.split('/')[-1].split('.')[0])
        
        if self.generate_target:
            if self.flip:
                self.flip_this_image = self.cfg.flip_this_image
            else:
                self.flip_this_image = False
        else:
            if self.flip and self.training:
                self.flip_this_image = np.random.randint(2) > 0.5
            else:
                self.flip_this_image = False

        calib = self.kitti_dataset.get_calibration(image_index)
        calib_R = self.kitti_dataset.get_right_calibration(image_index)
        
        if self.flip_this_image:
            calib, calib_R = calib_R, calib

        baseline = np.fabs(calib.P[0,3]-calib_R.P[0,3])/calib.P[0,0] # ~ 0.54
        t_cam2_from_cam0 = calib.t_cam2_from_cam0

        left_img = self.loader(left)
        right_img = self.loader(right)
        if not self.flip_this_image:
            dataL = self.dploader(disp_L)
        else:
            disp_R = disp_L[:-4] + '_r.npy'
            dataL = self.dploader(disp_R)

        # box labels
        if self.training or self.generate_target:
            if self.cfg.RPN3D_ENABLE:
                labels = self.kitti_dataset.get_label_objects(image_index)
                boxes, box3ds, ori_classes = get_kitti_annos(labels,
                    valid_classes= self.valid_classes)

                if len(boxes) > 0:
                    boxes[:, [2,3]] = boxes[:, [0,1]] + boxes[:, [2,3]]
                    boxes = clip_boxes(boxes, left_img.size, remove_empty=False)

                    # sort(far -> near)
                    inds = box3ds[:, 5].argsort()[::-1]
                    box3ds = box3ds[inds]
                    boxes = boxes[inds]
                    ori_classes = ori_classes[inds]

                    # sort by classes
                    inds = ori_classes.argsort(kind='stable')
                    box3ds = box3ds[inds]
                    boxes = boxes[inds]
                    ori_classes = ori_classes[inds]

                    boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
                    box3ds = torch.as_tensor(box3ds).reshape(-1, 7)

                    #transform to viewpoint from camera 
                    if self.cfg.learn_viewpoint:
                        h, w, l, x, y, z, alpha = torch.split(box3ds, [1,1,1,1,1,1,1], 1)
                        box3ds[:, 6:] = convert_to_viewpoint_torch(alpha, z, x)

                    target = Box3DList(boxes, left_img.size, mode="xyxy", box3d=box3ds, Proj=calib.P, Proj_R=calib_R.P)
                    classes = torch.as_tensor(ori_classes)
                    target.add_field('labels', classes)

                    if self.flip_this_image:
                        target = target.transpose(0)

                    if not self.flip_this_image:
                        save_file = '{}/{:06d}.npz'.format(self.save_path, image_index)
                        save_label_file = '{}/{:06d}_labels.npz'.format(self.save_path, image_index)
                    else:
                        save_file = '{}/{:06d}_flip.npz'.format(self.save_path, image_index)
                        save_label_file = '{}/{:06d}_flip_labels.npz'.format(self.save_path, image_index)

                    if self.generate_target:
                        locations = compute_locations_bev(self.cfg.Z_MIN, self.cfg.Z_MAX, self.cfg.VOXEL_Z_SIZE, 
                            self.cfg.X_MIN, self.cfg.X_MAX, self.cfg.VOXEL_X_SIZE, torch.device('cpu'))
                        xs, zs = locations[:, 0], locations[:, 1]

                        labels_maps = []
                        dist_bevs = []
                        ious = []

                        for cls in self.valid_classes:
                            ANCHORS_Y = self.cfg.RPN3D.ANCHORS_Y[cls-1]
                            ANCHORS_HEIGHT, ANCHORS_WIDTH, ANCHORS_LENGTH = self.cfg.RPN3D.ANCHORS_HEIGHT[cls-1], self.cfg.RPN3D.ANCHORS_WIDTH[cls-1], self.cfg.RPN3D.ANCHORS_LENGTH[cls-1]

                            ys = torch.zeros_like(xs) + ANCHORS_Y
                            locations3d = torch.stack([xs, ys, zs], dim=1)
                            locations3d = locations3d[:, None].repeat(1, self.cfg.num_angles, 1)

                            hwl = torch.as_tensor([ANCHORS_HEIGHT, ANCHORS_WIDTH, ANCHORS_LENGTH])
                            hwl = hwl[None, None].repeat(len(locations3d), self.cfg.num_angles, 1)

                            angles = torch.as_tensor(self.cfg.ANCHOR_ANGLES)
                            angles = angles[None].repeat(len(locations3d), 1)
                            sin, cos = torch.sin(angles), torch.cos(angles)

                            z_size, y_size, x_size = self.cfg.GRID_SIZE

                            anchors = torch.cat([hwl, locations3d, angles[:, :, None]], dim=2)
                            anchors[:, :, 4] += anchors[:, :, 0] / 2.
                            anchors = anchors.reshape(-1, 7)
                            anchors_boxlist = Box3DList(torch.zeros(len(anchors), 4), left_img.size, mode='xyxy', box3d=anchors, Proj=calib.P, Proj_R=calib_R.P)

                            inds_this_class = target.get_field('labels') == cls
                            target_box3ds = target.box3d[inds_this_class]

                            target_corners = (target.box_corners() + target.box3d[:, None, 3:6])[inds_this_class]
                            anchor_corners = anchors_boxlist.box_corners() + anchors_boxlist.box3d[:, None, 3:6]

                            dist_bev = torch.norm(anchor_corners[:, None, :4, [0,2]] - target_corners[None, :, :4, [0,2]], dim=-1)
                            dist_bev = dist_bev.mean(dim=-1)

                            dist_bev[dist_bev > 5.] = 5.
                            dist_bevs.append(dist_bev)

                            # note that this can one anchor <-> many labels
                            labels_map = torch.zeros((len(dist_bev), len(target_box3ds)), dtype=torch.uint8)
                            for i in range(len(target_box3ds)):
                                if (cls == 2 and self.less_car) or ((cls == 1 or cls == 3) and self.less_human):
                                    box_pixels = (target_box3ds[i, 1] * target_box3ds[i, 2]) / np.fabs(self.cfg.VOXEL_X_SIZE * self.cfg.VOXEL_Z_SIZE) / 4.
                                else:
                                    box_pixels = (target_box3ds[i, 1] * target_box3ds[i, 2]) / np.fabs(self.cfg.VOXEL_X_SIZE * self.cfg.VOXEL_Z_SIZE)
                                box_pixels = int(box_pixels)

                                topk_mindistance, topk_mindistance_ind = torch.topk(dist_bev[:, i], box_pixels, largest=False, sorted=False)

                                labels_map[topk_mindistance_ind[topk_mindistance < 5.], i] = cls
                            labels_maps.append(labels_map)

                        dist_bev = torch.cat(dist_bevs, dim=1)
                        sparse.save_npz(save_file, sparse.csr_matrix(dist_bev))
                        print('Saved {}'.format(save_file))
                        labels_map = torch.cat(labels_maps, dim=1)
                        sparse.save_npz(save_label_file, sparse.csr_matrix(labels_map))
                        print('Saved {}'.format(save_label_file))
                    else:
                        if self.training:
                            iou = sparse.load_npz(save_file)
                            labels_map = sparse.load_npz(save_label_file)

        w, h = left_img.size

        if self.flip_this_image:
            left_img, right_img = hflip(right_img), hflip(left_img)
            dataL = np.ascontiguousarray(dataL[:, ::-1])

        processed = preprocess.get_transform(augment=False)
        left_img = processed(left_img)
        right_img = processed(right_img)
        left_img = torch.reshape(left_img,[1,3,left_img.shape[1],left_img.shape[2]])
        right_img = torch.reshape(right_img,[1,3,right_img.shape[1],right_img.shape[2]])

        img_size = (left_img.shape[2], left_img.shape[3])

        top_pad = 384-left_img.shape[2]
        left_pad = 1248-left_img.shape[3]

        left_img = F.pad(left_img,(0,left_pad, 0,top_pad),'constant',0)
        right_img = F.pad(right_img,(0,left_pad, 0,top_pad),'constant',0)

        dataL = F.pad(torch.as_tensor(dataL), (0,left_pad, 0,top_pad), 'constant', -389.63037)

        outputs = [left_img, right_img, dataL, calib, calib_R]

        if self.training:
            outputs.append(image_index)
            if self.cfg.RPN3D_ENABLE:
                outputs.append(target)
            if self.cfg.RPN3D_ENABLE:
                outputs.append(iou)
                outputs.append(labels_map)
        else:
            outputs.extend([img_size, image_index])

        return outputs

    def __len__(self):
        return len(self.left)
 
