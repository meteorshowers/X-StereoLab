import torch
from torch.nn import functional as F
from torch import nn

from dsgn.utils.torch_utils import *

from functools import partial

from dsgn.layers.sigmoid_focal_loss import SigmoidFocalLoss
from dsgn.layers.iou_loss import IOULoss

INF = 100000000

def sigmoid_focal_loss_multi_target(logits, targets, weights=None, gamma=2., alpha=0.25):
    assert torch.all((targets == 1) | (targets == 0)), 'labels should be 0 or 1 in multitargetloss.'
    assert logits.shape == targets.shape
    t = targets
    p = torch.sigmoid(logits)
    term1 = (1 - p) ** gamma * torch.log(p + 1e-7)
    term2 = p ** gamma * torch.log(1 - p + 1e-7)
    loss = -(t == 1).float() * term1 * alpha - (t == 0).float() * term2 * (1 - alpha)
    if weights is None:
        return loss.sum()
    else:
        return (loss * weights).sum()

def smooth_l1_loss(input, target, weight, beta=1. / 9):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    return (loss.mean(dim=1) * weight).sum() / weight.sum()

class RPN3DLoss(object):

    def __init__(self, cfg):
        self.cfg = cfg

        self.cls_loss_func = partial(
            sigmoid_focal_loss_multi_target,
            gamma=self.cfg.RPN3D.FOCAL_GAMMA,
            alpha=self.cfg.RPN3D.FOCAL_ALPHA)
        self.box_reg_loss_func = smooth_l1_loss #IOULoss()
        self.centerness_loss_func = nn.BCEWithLogitsLoss()
        self.num_angles = self.cfg.num_angles
        self.num_classes = self.cfg.num_classes
        self.anchors_y = torch.as_tensor(self.cfg.RPN3D.ANCHORS_Y)
        self.anchor_angles = torch.as_tensor(self.cfg.ANCHOR_ANGLES)
        self.centerness4class = getattr(self.cfg, 'centerness4class', False)
        
        self.norm_expdist = getattr(self.cfg, 'norm_expdist', False)

        self.valid_classes = getattr(self.cfg, 'valid_classes', None)
        self.class4angles = getattr(self.cfg, 'class4angles', True)
        self.norm_factor = getattr(self.cfg, 'norm_factor', 1.)
        self.norm_max = getattr(self.cfg, 'norm_max', False)
        self.box_corner_parameters = getattr(self.cfg, 'box_corner_parameters', True)

        self.pred_reg_dim = 24 if self.box_corner_parameters else 7
        self.target_reg_dim = (4 + 24) if self.box_corner_parameters else (4 + 7)

        self.CV_X_MIN, self.CV_Y_MIN, self.CV_Z_MIN = cfg.CV_X_MIN, cfg.CV_Y_MIN, cfg.CV_Z_MIN
        self.CV_X_MAX, self.CV_Y_MAX, self.CV_Z_MAX = cfg.CV_X_MAX, cfg.CV_Y_MAX, cfg.CV_Z_MAX
        self.X_MIN, self.Y_MIN, self.Z_MIN = cfg.X_MIN, cfg.Y_MIN, cfg.Z_MIN
        self.X_MAX, self.Y_MAX, self.Z_MAX = cfg.X_MAX, cfg.Y_MAX, cfg.Z_MAX
        self.VOXEL_X_SIZE, self.VOXEL_Y_SIZE, self.VOXEL_Z_SIZE = cfg.VOXEL_X_SIZE, cfg.VOXEL_Y_SIZE, cfg.VOXEL_Z_SIZE

    def prepare_targets(self, locations, targets, ious=None, labels_map=None):
        labels, reg_targets = [], []
        labels_centerness = []

        xs, zs = locations[:, 0], locations[:, 1]

        ys_cls = torch.zeros_like(xs)[:,None] + self.anchors_y.cuda()[None]
        xs_cls = xs[:,None].repeat(1, self.num_classes)
        zs_cls = zs[:,None].repeat(1, self.num_classes)
        for i in range(len(targets)):
            target = targets[i]
            labels_per_im = target.get_field('labels')

            non_ign_inds = labels_per_im.float() < 3.5 # note it should not be labels_per_im < 3.5

            if non_ign_inds.sum() > 0:
                iou = torch.as_tensor(ious[i].toarray()).cuda()
                labels_precomputed = torch.as_tensor(labels_map[i].toarray())

                box3ds = target.box3d[non_ign_inds].cuda()
                box3ds_corners = (target.box_corners() + target.box3d[:, None, 3:6])[non_ign_inds].cuda()
                labels_per_im = labels_per_im[non_ign_inds].cuda()

                box3ds_rect_bev = box3ds_corners[:, :4, [0,2]]
                box3ds_rect_bev = torch.cat([box3ds_rect_bev.min(1)[0], box3ds_rect_bev.max(1)[0]], dim=1)

                box3ds_centers = box3ds_corners.mean(dim=1)

                l = xs[:, None] - box3ds_rect_bev[:, 0][None]
                t = zs[:, None] - box3ds_rect_bev[:, 1][None]
                r = box3ds_rect_bev[:, 2][None] - xs[:, None]
                b = box3ds_rect_bev[:, 3][None] - zs[:, None]

                reg_targets2d_per_im = torch.stack([l, t, r, b], dim=2)
                reg_targets2d_per_im = reg_targets2d_per_im[:, None, :, :].repeat(1, self.num_classes, 1, 1).reshape(
                    len(locations), self.num_classes, len(box3ds), 4)

                locations3d = torch.stack([xs_cls, ys_cls, zs_cls], dim=2).reshape(-1, 3)

                if not self.box_corner_parameters:
                    reg_targets_per_im = (box3ds_centers[None] - locations3d[:,None]).reshape(len(locations), self.num_classes, len(box3ds), 3)
                    box3ds_parameters = torch.cat([box3ds[:, 0:3], box3ds[:, 6:]], dim=1)
                    reg_targets_per_im = torch.cat([reg_targets_per_im, box3ds_parameters[None, None, :].expand(len(locations), self.num_classes, len(box3ds), 4)], dim=3)
                else:
                    reg_targets_per_im = (box3ds_corners[None] - locations3d[:,None,None]).reshape(len(locations), self.num_classes, len(box3ds), 24)

                reg_targets_per_im = torch.cat([reg_targets2d_per_im, reg_targets_per_im], dim=-1)

                assert iou.shape[1] == len(box3ds), 'Number of Pre computed iou does not match current gts.'
                locations_min_dist, locations_gt_inds = iou.min(dim=1)

                labels_precomputed_inverse = -1 + torch.zeros((len(labels_precomputed), 1 + self.num_classes), dtype=torch.int32, device='cuda')
                labels_precomputed_inverse.scatter_(1, labels_precomputed.long().cuda(), 
                    torch.arange(len(box3ds))[None].expand(len(labels_precomputed), len(box3ds)).int().cuda() )
                labels_precomputed_inverse = labels_precomputed_inverse[:, 1:]
                labels_precomputed_inverse = labels_precomputed_inverse.reshape(-1, self.num_angles, self.num_classes)

                labels_per_im = (labels_precomputed_inverse >= 0).int()

                if self.norm_expdist:
                    min_dists = []
                    max_dists = []
                    for i in range(iou.shape[1]):
                        if labels_precomputed[:, i].sum() == 0:
                            max_dist = 1.
                            max_dists.append(max_dist)
                            min_dist = 0.
                            min_dists.append(min_dist)
                        else:
                            min_dist = iou[labels_precomputed[:, i] > 0, i].min().clamp(max=5.)
                            min_dists.append(min_dist)
                            if self.norm_max:
                                max_dist = iou[labels_precomputed[:, i] > 0, i].max().clamp(min=0.)
                                max_dists.append(max_dist)

                    min_dists = torch.as_tensor(min_dists, device='cuda')
                    if self.norm_max:
                        max_dists = torch.as_tensor(max_dists, device='cuda')
                        locations_norm_min_dist = (iou - min_dists[None]) / (max_dists[None] - min_dists[None]) 
                    else:
                        locations_norm_min_dist = iou - min_dists[None]
                    
                    if not self.centerness4class:
                        labels_centerness_per_im = locations_norm_min_dist[range(len(iou)), locations_gt_inds]
                        labels_centerness_per_im = labels_centerness_per_im.reshape(-1, self.num_angles)
                    else:
                        labels_centerness_per_im = locations_norm_min_dist.gather(1, 
                            (labels_precomputed_inverse * (labels_precomputed_inverse > 0).int()).reshape(-1, self.num_classes).long())
                        labels_centerness_per_im = labels_centerness_per_im.reshape(-1, self.num_angles, self.num_classes)
                    labels_centerness_per_im = torch.exp(-labels_centerness_per_im * self.norm_factor)
                else:
                    labels_centerness_per_im = torch.exp(-locations_min_dist)

                reg_targets_per_im = reg_targets_per_im[:, None].repeat(1, self.num_angles, 1, 1, 1).reshape(-1, len(box3ds), self.target_reg_dim)
                reg_targets_per_im = reg_targets_per_im[torch.arange(len(reg_targets_per_im)), labels_precomputed_inverse.reshape(-1).long()]
                reg_targets_per_im = reg_targets_per_im.reshape(-1, self.num_angles, self.num_classes, self.target_reg_dim)
            else:
                labels_per_im = torch.zeros(len(locations), self.num_angles, self.num_classes, dtype=torch.int32).cuda()
                reg_targets_per_im = torch.zeros(len(locations), self.num_angles, self.num_classes, self.target_reg_dim, dtype=torch.float32).cuda()
                if not self.centerness4class:
                    labels_centerness_per_im = torch.zeros(len(locations), self.num_angles, dtype=torch.float32).cuda()
                else:
                    labels_centerness_per_im = torch.zeros(len(locations), self.num_angles, self.num_classes, dtype=torch.float32).cuda()

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)
            labels_centerness.append(labels_centerness_per_im)

        return labels, reg_targets, labels_centerness

    def __call__(self, bbox_cls, bbox_reg, bbox_centerness, targets, calib, calib_R, 
        ious=None, labels_map=None):
        N = bbox_cls.shape[0]

        locations_bev = compute_locations_bev(self.Z_MIN, self.Z_MAX, self.VOXEL_Z_SIZE, 
            self.X_MIN, self.X_MAX, self.VOXEL_X_SIZE, bbox_cls.device)

        labels, reg_targets, labels_centerness = self.prepare_targets(locations_bev, targets, ious=ious, labels_map=labels_map)

        labels = torch.stack(labels, dim=0)
        reg_targets = torch.stack(reg_targets, dim=0)
        labels_centerness = torch.stack(labels_centerness, dim=0)

        if self.class4angles:
            bbox_cls = bbox_cls.reshape(N, self.num_angles * self.num_classes, -1).transpose(1, 2).reshape(N, -1, self.num_angles, self.num_classes)
        else:
            bbox_cls = bbox_cls.reshape(N, self.num_classes, -1).transpose(1, 2).reshape(N, -1, self.num_classes)

        if not self.centerness4class:
            bbox_centerness = bbox_centerness.reshape(N, self.num_angles, -1).transpose(1, 2)
        else:
            bbox_centerness = bbox_centerness.reshape(N, self.num_angles * self.num_classes, -1).transpose(1, 2)

        bbox_reg = bbox_reg.reshape(N, self.num_angles * self.num_classes * self.pred_reg_dim, -1).transpose(1, 2).reshape(N, -1, self.num_angles, self.num_classes, self.pred_reg_dim)

        loss = 0.
        cls_loss = 0.
        reg_loss = 0.
        centerness_loss = 0.

        if self.class4angles:
            bbox_cls = bbox_cls.reshape(-1, self.num_angles * self.num_classes)
        else:
            bbox_cls = bbox_cls.reshape(-1, self.num_classes)
        labels = labels.reshape(-1, self.num_angles * self.num_classes)
        bbox_reg = bbox_reg.reshape(-1, self.num_angles * self.num_classes, self.pred_reg_dim)
        reg_targets = reg_targets.reshape(-1, self.num_angles * self.num_classes, self.target_reg_dim)
        if not self.centerness4class:
            bbox_centerness = bbox_centerness.reshape(-1, self.num_angles)
            labels_centerness = labels_centerness.reshape(-1, self.num_angles)
        else:
            bbox_centerness = bbox_centerness.reshape(-1, self.num_angles * self.num_classes)
            labels_centerness = labels_centerness.reshape(-1, self.num_angles * self.num_classes)

        # cls loss
        pos_inds = torch.nonzero(labels > 0)

        if self.class4angles:
            labels_class = labels
        else:
            labels_class = labels.reshape(-1, self.num_angles, self.num_classes).sum(dim=1) > 0

        cls_loss += self.cls_loss_func(
            bbox_cls,
            labels_class.int()
        ) / (pos_inds.shape[0] + 10)  # add N to avoid dividing by a zero

        bbox_reg = bbox_reg[pos_inds[:, 0], pos_inds[:, 1]]
        reg_targets = reg_targets[pos_inds[:, 0], pos_inds[:, 1]]
        if not self.centerness4class:
            labels_centerness = labels_centerness[pos_inds[:, 0], pos_inds[:, 1] // self.num_classes]
        else:
            labels_centerness = labels_centerness[pos_inds[:, 0], pos_inds[:, 1]]

        reg_targets_theta = reg_targets[:, -1:]
        bbox_reg_theta = bbox_reg[:, -1:]

        reg_targets = torch.cat([reg_targets[:, :-1], torch.sin(reg_targets_theta * 0.5) * torch.cos(bbox_reg_theta * 0.5)], dim=1)
        bbox_reg = torch.cat([bbox_reg[:, :-1], torch.cos(reg_targets_theta * 0.5) * torch.sin(bbox_reg_theta * 0.5)], dim=1)

        if not self.centerness4class:
            bbox_centerness = bbox_centerness[pos_inds[:, 0], pos_inds[:, 1] // self.num_classes]
        else:
            bbox_centerness = bbox_centerness[pos_inds[:, 0], pos_inds[:, 1]]

        # reg loss
        if pos_inds.shape[0] > 0:
            box2d_targets, box3d_corners_targets = torch.split(reg_targets, [4, self.pred_reg_dim], dim=1)
            centerness_targets = labels_centerness

            reg_loss += self.box_reg_loss_func(
                bbox_reg,
                box3d_corners_targets,
                centerness_targets
            )
            centerness_loss += self.centerness_loss_func(
                bbox_centerness,
                centerness_targets
            )
        else:
            reg_loss += bbox_reg.sum()
            centerness_loss += bbox_centerness.sum()

        loss = cls_loss + reg_loss + centerness_loss

        return loss, cls_loss, reg_loss, centerness_loss
