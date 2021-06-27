""" Helper methods for loading and parsing KITTI data.

Author: Charles R. Qi
Date: September 2017
"""
from __future__ import print_function

import numpy as np
import cv2
import os

class Object3d(object):
    ''' 3d object label '''
    def __init__(self, label_file_line):
        data = label_file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]

        # extract label, truncation, occlusion
        self.type = data[0] # 'Car', 'Pedestrian', ...
        self.truncation = data[1] # truncated pixel ratio [0..1]
        self.occlusion = int(data[2]) # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3] # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4] # left
        self.ymin = data[5] # top
        self.xmax = data[6] # right
        self.ymax = data[7] # bottom
        
        # extract 3d bounding box information
        self.h = data[8] # box height
        self.w = data[9] # box width
        self.l = data[10] # box length (in meters)
        self.cx = data[11] # x in camera coord
        self.cy = data[12] # y in camera coord
        self.cz = data[13] # z in camera coord
        self.t = (data[11],data[12],data[13]) # location (x,y,z) in camera coord.
        self.ry = data[14] # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

        self.score = None
        if len(data) == 16:
            self.score = data[15]

    @property
    def box2d(self):
        if not hasattr(self, '_box2d'):
            self._box2d = np.array([self.xmin,self.ymin,self.xmax,self.ymax])
        return self._box2d

    @box2d.setter
    def box2d(self, var):
        self.xmin,self.ymin,self.xmax,self.ymax = var
        self._box2d = var

    @property
    def box3d(self):
        # x, y, z, h, w, l, ry 
        # rect camera coordinate
        if not hasattr(self, '_box3d'):
            self._box3d = np.array([self.cx, self.cy, self.cz, self.h, self.w, self.l, self.ry])
        return self._box3d

    @box3d.setter
    def box3d(self, var):
        self.cx, self.cy, self.cz, self.h, self.w, self.l, self.ry = var
        self.t = (self.cx, self.cy, self.cz)
        self._box3d = var

    @classmethod
    def fromfile(cls, calib_filepath, from_video=False):
        if from_video:
            calibs = cls.read_calib_from_video(calib_filepath)
        else:
            calibs = cls.read_calib_file(calib_filepath)
        P = calibs['P2'] 
        V2C = calibs['Tr_velo_to_cam']
        R0 = calibs['R0_rect']
        return cls(P, V2C, R0)

    def print_object(self):
        print('Type, truncation, occlusion, alpha: %s, %d, %d, %f' % \
            (self.type, self.truncation, self.occlusion, self.alpha))
        print('2d bbox (x0,y0,x1,y1): %f, %f, %f, %f' % \
            (self.xmin, self.ymin, self.xmax, self.ymax))
        print('3d bbox h,w,l: %f, %f, %f' % \
            (self.h, self.w, self.l))
        print('3d bbox location, ry: (%f, %f, %f), %f' % \
            (self.t[0],self.t[1],self.t[2],self.ry))

    def dumpstr(self):
        if self.score:
            return '%s %f %d %f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f' % (self.type, self.truncation, self.occlusion, self.alpha,
                self.xmin, self.ymin, self.xmax, self.ymax, 
                self.h, self.w, self.l, self.cx, self.cy, self.cz, self.ry,
                self.score)
        else:
            return '%s %f %d %f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f' % (self.type, self.truncation, self.occlusion, self.alpha,
                self.xmin, self.ymin, self.xmax, self.ymax, 
                self.h, self.w, self.l, self.cx, self.cy, self.cz, self.ry)

    def __str__(self):
        return self.dumpstr()

class Calibration(object):
    ''' Calibration matrices and utils
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.

        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref

        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]

        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)

        velodyne coord:
        front x, left y, up z

        rect/ref camera coord:
        right x, down y, front z

        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf

        TODO(rqi): do matrix multiplication only once for each projection.
    '''
    def __init__(self, P, V2C, R0, t_cam2_from_cam0=None):
        # Projection matrix from rect camera coord to image2 coord (left color image carmera)
        self.P = P
        self.P = np.reshape(self.P, [3,4])
        # Rigid transform from Velodyne coord to reference camera coord
        self.V2C = V2C
        self.V2C = np.reshape(self.V2C, [3,4])
        self.C2V = inverse_rigid_trans(self.V2C)
        # Rotation from reference camera coord to rect camera coord
        self.R0 = R0
        self.R0 = np.reshape(self.R0,[3,3])

        # Camera intrinsics and extrinsics
        self.c_u = self.P[0,2]
        self.c_v = self.P[1,2]
        self.f_u = self.P[0,0]
        self.f_v = self.P[1,1]
        self.b_x = self.P[0,3]/(-self.f_u) # relative 
        self.b_y = self.P[1,3]/(-self.f_v)

        # default camera 2 
        # 3D coordinate is relative to camera 0
        # thus it needs to be translated to camera 2
        self.t_cam2_from_cam0 = t_cam2_from_cam0

    @classmethod
    def fromfile(cls, calib_filepath, from_video=False):
        if from_video:
            calibs = cls.read_calib_from_video(calib_filepath)
        else:
            calibs = cls.read_calib_file(calib_filepath)
        P = calibs['P2'] 
        V2C = calibs['Tr_velo_to_cam']
        R0 = calibs['R0_rect']

        t_cam2_from_cam0 = (calibs['P2'][3] - calibs['P0'][3]) / P[0]
        return cls(P, V2C, R0, t_cam2_from_cam0=t_cam2_from_cam0)

    @classmethod
    def fromrightfile(cls, calib_filepath, from_video=False):
        if from_video:
            calibs = cls.read_calib_from_video(calib_filepath)
        else:
            calibs = cls.read_calib_file(calib_filepath)
        P = calibs['P3'] 
        V2C = calibs['Tr_velo_to_cam']
        R0 = calibs['R0_rect']

        t_cam2_from_cam0 = (calibs['P2'][3] - calibs['P0'][3]) / P[0]
        return cls(P, V2C, R0, t_cam2_from_cam0=t_cam2_from_cam0)

    @classmethod
    def default(cls):
        P = np.array([7.070493000000e+02, 0.000000000000e+00, 6.040814000000e+02, 4.575831000000e+01, 
                      0.000000000000e+00, 7.070493000000e+02, 1.805066000000e+02, -3.454157000000e-01, 
                      0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 4.981016000000e-03]).reshape(3,4)

        V2C = np.array([
            [0., -1., 0., 0.],
            [0., 0., -1., 0.],
            [1., 0., 0., 0.],
            ])
        R0 = np.array([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]
            ])
        return cls(P, V2C, R0)

    @classmethod
    def read_calib_file(self, filepath):
        ''' Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        '''
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line)==0: continue
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        return data
    
    @classmethod
    def read_calib_from_video(self, calib_root_dir):
        ''' Read calibration for camera 2 from video calib files.
            there are calib_cam_to_cam and calib_velo_to_cam under the calib_root_dir
        '''
        data = {}
        cam2cam = self.read_calib_file(os.path.join(calib_root_dir, 'calib_cam_to_cam.txt'))
        velo2cam = self.read_calib_file(os.path.join(calib_root_dir, 'calib_velo_to_cam.txt'))
        Tr_velo_to_cam = np.zeros((3,4))
        Tr_velo_to_cam[0:3,0:3] = np.reshape(velo2cam['R'], [3,3])
        Tr_velo_to_cam[:,3] = velo2cam['T']
        data['Tr_velo_to_cam'] = np.reshape(Tr_velo_to_cam, [12])
        data['R0_rect'] = cam2cam['R_rect_00']
        data['P2'] = cam2cam['P_rect_02']
        return data

    def cart2hom(self, pts_3d):
        ''' Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        '''
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n,1))))
        return pts_3d_hom
 
    # =========================== 
    # ------- 3d to 3d ---------- 
    # =========================== 
    def project_velo_to_ref(self, pts_3d_velo):
        pts_3d_velo = self.cart2hom(pts_3d_velo) # nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2C))

    def project_ref_to_velo(self, pts_3d_ref):
        pts_3d_ref = self.cart2hom(pts_3d_ref) # nx4
        return np.dot(pts_3d_ref, np.transpose(self.C2V))

    def project_rect_to_ref(self, pts_3d_rect):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))
    
    def project_ref_to_rect(self, pts_3d_ref):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))
 
    def project_rect_to_velo(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx3 points in velodyne coord.
        ''' 
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    def project_velo_to_rect(self, pts_3d_velo):
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)

    # =========================== 
    # ------- 3d to 2d ---------- 
    # =========================== 
    def project_rect_to_image(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P)) # nx3
        pts_2d[:,0] /= pts_2d[:,2]
        pts_2d[:,1] /= pts_2d[:,2]
        return pts_2d[:,0:2]
        
    def project_velo_to_image(self, pts_3d_velo):
        ''' Input: nx3 points in velodyne coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect)

    # =========================== 
    # ------- 2d to 3d ---------- 
    # =========================== 
    def project_image_to_rect(self, uv_depth):
        ''' Input: nx3 first two channels are uv, 3rd channel
                   is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        '''
        n = uv_depth.shape[0]
        x = ((uv_depth[:,0]-self.c_u)*uv_depth[:,2])/self.f_u + self.b_x
        y = ((uv_depth[:,1]-self.c_v)*uv_depth[:,2])/self.f_v + self.b_y
        pts_3d_rect = np.zeros((n,3))
        pts_3d_rect[:,0] = x
        pts_3d_rect[:,1] = y
        pts_3d_rect[:,2] = uv_depth[:,2]
        return pts_3d_rect

    def project_image_to_velo(self, uv_depth):
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return self.project_rect_to_velo(pts_3d_rect)

    ### process boxes 3d
    def boxes3d_project_rect_to_velo(self, boxes3d):
        ''' Input: nx7 boxes3d in rect camera coord
                    x, y, z, h, w, l, ry
            Output: nx7 boxes3d in velodyne coord
            Note that only rotation and translation is applied, thus the [h, w, l] is same.
        '''
        ry = boxes3d[:,6:]
        rz = format_angles(-ry-np.pi/2)
        boxes3d_velo = np.concatenate(
            [self.project_rect_to_velo(boxes3d[:,:3]), boxes3d[:, 3:6], rz], axis=1)
        return boxes3d_velo
    
    def boxes3d_project_velo_to_rect(self, boxes3d):
        ''' Input: nx7 boxes3d in velodyne coord
                    x, y, z, h, w, l, rz
            Output: nx7 boxes3d in rect camera coord
            Note that only rotation and translation is applied, thus the [h, w, l] is same.
        '''
        rz = boxes3d[:,6:]
        ry = format_angles(-rz-np.pi/2)
        boxes3d_rect = np.concatenate(
            [self.project_velo_to_rect(boxes3d[:,:3]), boxes3d[:, 3:6], ry], axis=1)
        return boxes3d_rect

def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr) # 3x4
    inv_Tr[0:3,0:3] = np.transpose(Tr[0:3,0:3])
    inv_Tr[0:3,3] = np.dot(-np.transpose(Tr[0:3,0:3]), Tr[0:3,3])
    return inv_Tr

def read_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    objects = [Object3d(line) for line in lines]
    return objects

def load_image(img_filename):
    return cv2.imread(img_filename)

def load_velo_scan(velo_filename):
    scan = np.fromfile(velo_filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan
