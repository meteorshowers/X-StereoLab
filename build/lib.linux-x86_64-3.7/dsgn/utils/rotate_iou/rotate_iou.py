import ctypes
import time
import numpy as np
import os.path as osp
ll = ctypes.cdll.LoadLibrary 

CUR_DIR = osp.dirname(osp.abspath(__file__))
lib = ll(CUR_DIR+"/calculate_iou.so") 
lib.calculate_iou.restype = ctypes.c_double

# Prepare for using in C++
def change_to_array(input_list):
    result_arr2=ctypes.c_double*2
    result_arr42 = result_arr2*4
    result_arr = result_arr42()
    for i in range(4):
        result_arr[i][0] = input_list[i][0]
        result_arr[i][1] = input_list[i][1]
    return result_arr


# The main function for calculating the IoU
# Return in double precision
def rotate_iou(corner1, corner2):
    # input_arr_1 = change_to_array(corner1)
    # input_arr_2 = change_to_array(corner2)
    # iou = lib.calculate_iou(input_arr_1, input_arr_2)
    iou = lib.calculate_iou(
        ctypes.c_double(corner1[0,0]), ctypes.c_double(corner1[0,1]), ctypes.c_double(corner1[1,0]), ctypes.c_double(corner1[1,1]), ctypes.c_double(corner1[2,0]), ctypes.c_double(corner1[2,1]), ctypes.c_double(corner1[3,0]), ctypes.c_double(corner1[3,1]),
        ctypes.c_double(corner2[0,0]), ctypes.c_double(corner2[0,1]), ctypes.c_double(corner2[1,0]), ctypes.c_double(corner2[1,1]), ctypes.c_double(corner2[2,0]), ctypes.c_double(corner2[2,1]), ctypes.c_double(corner2[3,0]), ctypes.c_double(corner2[3,1]))
    return iou

def compute_iou_fast(boxes1, boxes2, corner_boxes1=None, corner_boxes2=None):
    """
        boxes2 should be less than boxes1 in order to speed up
    """
    if corner_boxes1 is None or corner_boxes2 is None:
        from .utils import boxes_center2d_to_corner2d_lidar
    from functools import partial
    max_lens_1 = np.linalg.norm(boxes1[:, [2,3]], axis=1) / 2
    max_lens_2 = np.linalg.norm(boxes2[:, [2,3]], axis=1) / 2
    if corner_boxes1 is None: corner_boxes1 = boxes_center2d_to_corner2d_lidar(boxes1)
    if corner_boxes2 is None: corner_boxes2 = boxes_center2d_to_corner2d_lidar(boxes2)
    iou = np.zeros((len(boxes1), len(boxes2)))
    for i in range(len(boxes2)):
        corner_iou = partial(rotate_iou, corner2=corner_boxes2[i])
        near_boxes = np.linalg.norm(boxes1[:,:2] - boxes2[i,:2], axis=1) <= max_lens_1 + max_lens_2[i]
        near_iou = np.asarray(list(map(corner_iou, corner_boxes1[near_boxes])))
        iou[near_boxes, i] = near_iou
    return iou

compute_iou_fast_2dboxes = compute_iou_fast

def compute_2diou_fast_3d(boxes1, boxes2, corner_boxes1=None, corner_boxes2=None):
    assert boxes1.shape[1] == 7 and boxes2.shape[1] == 7, '3D boxes take 7 parameters.'
    if corner_boxes1 is not None: corner_boxes1 = corner_boxes1[:, :4, :2]
    if corner_boxes2 is not None: corner_boxes2 = corner_boxes2[:, :4, :2]
    return compute_iou_fast_2dboxes(boxes1[:, [0, 1, 4, 5, 6]], boxes2[:, [0, 1, 4, 5, 6]], corner_boxes1, corner_boxes2)

def iou_3d(corner1, corner2):
    iou2d = rotate_iou(corner1, corner2)
    area1 = np.linalg.norm( corner1[0, :2] - corner1[1, :2] ) * np.linalg.norm( corner1[0, :2] - corner1[2, :2] )
    area2 = np.linalg.norm( corner2[0, :2] - corner2[1, :2] ) * np.linalg.norm( corner2[0, :2] - corner2[2, :2] )
    height1 = corner1[4, 2] - corner1[0, 2]
    height2 = corner2[4, 2] - corner2[0, 2]
    inter_area = (area1 + area2) * iou2d / (1. + iou2d)
    zmax = min(corner1[4, 2], corner2[4, 2])
    zmin = max(corner1[0, 2], corner2[0, 2])
    inter_vol = inter_area * max(0., zmax-zmin)
    iou3d = inter_vol / (area1 * height1 + area2 * height2 - inter_vol)
    return iou3d

