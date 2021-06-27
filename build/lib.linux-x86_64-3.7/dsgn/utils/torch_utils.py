import numpy as np

import torch

def project_image_to_rect(uv_depth, P):
    # uv_depth (3, N)

    c_u = P[0,2]
    c_v = P[1,2]
    f_u = P[0,0]
    f_v = P[1,1]
    b_x = P[0,3]/(-f_u) # relative 
    b_y = P[1,3]/(-f_v)

    # use camera coordinate
    n = uv_depth.shape[1]
    x = ((uv_depth[0]-c_u)*uv_depth[2])/f_u + b_x
    y = ((uv_depth[1]-c_v)*uv_depth[2])/f_v + b_y
    return torch.stack([x, y, uv_depth[2]], dim=0)

def project_disp_to_depth_new(points_cam, Proj):
    xs, ys, disp = points_cam[0:1], points_cam[1:2], points_cam[2:3]
    _, h, w, d = disp.shape

    baseline = 0.54
    mask = disp > 0
    depth = Proj[0,0] * baseline / (disp + 1. - mask.float())
    points = torch.cat([xs, ys, depth], dim=0)
    points = points.reshape((3, -1))

    # camera coordinate
    cloud = project_image_to_rect(points, Proj)
    cloud = cloud.reshape(3, h, w, d)
    return cloud

def project_rect_to_image(pts_3d_rect, P):
    n = pts_3d_rect.shape[0]
    ones = torch.ones((n,1))
    if pts_3d_rect.is_cuda:
        ones = ones.cuda()
    pts_3d_rect = torch.cat([pts_3d_rect, ones], dim=1)
    pts_2d = torch.mm(pts_3d_rect, torch.transpose(P, 0, 1)) # nx3
    pts_2d[:,0] /= pts_2d[:,2]
    pts_2d[:,1] /= pts_2d[:,2]
    return pts_2d[:,0:2]

# def compute_locations(h, w, stride, device):
#     shifts_x = torch.arange(
#         0, w * stride, step=stride,
#         dtype=torch.float32, device=device
#     )
#     shifts_y = torch.arange(
#         0, h * stride, step=stride,
#         dtype=torch.float32, device=device
#     )
#     shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
#     shift_x = shift_x.reshape(-1)
#     shift_y = shift_y.reshape(-1)
#     locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
#     return locations

# def compute_locations_3d(h, w, stride, device):
#     shifts_x = torch.arange(
#         0, w * stride, step=stride,
#         dtype=torch.float32, device=device
#     )
#     shifts_y = torch.arange(
#         0, h * stride, step=stride,
#         dtype=torch.float32, device=device
#     )
#     shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
#     shift_x = shift_x.reshape(-1)
#     shift_y = shift_y.reshape(-1)
#     locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
#     return locations

def compute_locations_bev(Z_MIN, Z_MAX, VOXEL_Z_SIZE, X_MIN, X_MAX, VOXEL_X_SIZE, device):
    shifts_z = torch.arange(Z_MIN, Z_MAX - np.sign(VOXEL_Z_SIZE) * 1e-10, step=VOXEL_Z_SIZE, 
        dtype=torch.float32).to(device) + VOXEL_Z_SIZE / 2.
    shifts_x = torch.arange(X_MIN, X_MAX - np.sign(VOXEL_X_SIZE) * 1e-10, step=VOXEL_X_SIZE,
        dtype=torch.float32).to(device) + VOXEL_X_SIZE / 2.
    shifts_z, shifts_x = torch.meshgrid(shifts_z, shifts_x)
    locations_bev = torch.stack([shifts_x, shifts_z], dim=-1)
    locations_bev = locations_bev.reshape(-1, 2)
    return locations_bev

def compute_centerness_targets(reg_targets):
    left_right = reg_targets[:, [0, 2]]
    top_bottom = reg_targets[:, [1, 3]]
    centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                  (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
    return torch.sqrt(centerness)

# def compute_corners_R(dimensions, rot):
#     num_boxes = dimensions.shape[0]
#     h, w, l = torch.split( dimensions.view(num_boxes, 1, 3), [1, 1, 1], dim=2)
#     # zeros = torch.zeros((num_boxes, 1, 1), dtype=torch.float32).cuda()
#     corners = torch.cat([torch.cat([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2], dim=2),
#                          torch.cat([h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2], dim=2),
#                          torch.cat([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2], dim=2)], dim=1)
#     corners = torch.matmul(rot, corners)
#     return corners

# def angle_to_bevrotation(sin, cos):
#     zeros = torch.zeros_like(sin)
#     rot_1 = torch.stack([cos, zeros, -sin], dim=1)
#     rot_2 = torch.stack([zeros, zeros+1., zeros], dim=1)
#     rot_3 = torch.stack([sin, zeros, cos], dim=1)
#     rot = torch.stack([rot_1, rot_2, rot_3], dim=1)
#     return rot

def convert_to_viewpoint_torch(ry, z, x):
    return ry + torch.atan2(z, x) - np.pi / 2

def convert_to_ry_torch(alpha, z, x):
    return alpha - torch.atan2(z, x) + np.pi / 2
