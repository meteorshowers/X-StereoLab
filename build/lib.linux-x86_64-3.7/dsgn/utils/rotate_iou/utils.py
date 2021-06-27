import numpy as np

def boxes_center3d_to_corner3d_lidar(boxes_center):
    # (N, 7) -> (N, 8, 3)
    N = boxes_center.shape[0]

    translation = boxes_center[:, :3]
    size = boxes_center[:, 3:6]
    rotation = boxes_center[:, 6]
    h, w, l = boxes_center[:,3], boxes_center[:,4], boxes_center[:,5]
    zeros = np.zeros((len(h)), dtype=np.float32)

    # N,8
    trackletBox_l = np.stack([-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2], axis=1)
    trackletBox_w = np.stack([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], axis=1)
    trackletBox_h = np.stack([zeros, zeros, zeros, zeros, h, h, h, h], axis=1)

    trackletBox = np.stack([trackletBox_l, trackletBox_w, trackletBox_h], axis=1)

    rotMat = np.zeros((N, 3, 3), dtype=np.float32)
    rotMat[:, 0, 0] = np.cos(rotation)
    rotMat[:, 0, 1] = -np.sin(rotation)
    rotMat[:, 1, 0] = np.sin(rotation)
    rotMat[:, 1, 1] = np.cos(rotation)
    rotMat[:, 2, 2] = 1.

    # N, 3, 8
    corner = np.matmul(rotMat, trackletBox) + translation[..., np.newaxis]
    corner = np.transpose(corner, (0, 2, 1))

    return corner

def boxes_center2d_to_corner2d_lidar(boxes_center):
    N = boxes_center.shape[0]
    boxes3d_center = np.zeros((N, 7))
    boxes3d_center[:, [0, 1, 4, 5, 6]] = boxes_center
    corner = boxes_center3d_to_corner3d_lidar(boxes3d_center)
    return corner[:, :4, :2]

