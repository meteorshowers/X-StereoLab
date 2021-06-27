from __future__ import print_function

from .submodule import *
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math

from dsgn.utils.bounding_box import compute_corners, quan_to_angle, \
    angle_to_quan, quan_to_rotation, compute_corners_sc
from dsgn.layers import BuildCostVolume

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

class StereoNet(nn.Module):
    def __init__(self, cfg=None):
        super(StereoNet, self).__init__()
        self.maxdisp = cfg.maxdisp
        self.downsample_disp = cfg.downsample_disp
        self.cfg = cfg
        self.num_classes = self.cfg.num_classes
        self.hg_rpn_conv3d = getattr(self.cfg, 'hg_rpn_conv3d', False)
        self.hg_rpn_conv = getattr(self.cfg, 'hg_rpn_conv', False)
        self.centerness4class = getattr(self.cfg, 'centerness4class', False)
        self.img_feature_attentionbydisp = getattr(self.cfg, 'img_feature_attentionbydisp', False)
        self.voxel_attentionbydisp = getattr(self.cfg, 'voxel_attentionbydisp', False)
        self.valid_classes = getattr(self.cfg, 'valid_classes', None)
        self.class4angles = getattr(self.cfg, 'class4angles', True)
        self.box_corner_parameters = getattr(self.cfg, 'box_corner_parameters', True)
        self.PlaneSweepVolume = getattr(self.cfg, 'PlaneSweepVolume', True)
        self.loss_disp = getattr(self.cfg, 'loss_disp', True)
        self.fix_centerness_bug = getattr(self.cfg, 'fix_centerness_bug', False)
        self.hg_firstconv = getattr(self.cfg, 'hg_firstconv', False)
        self.rpn3d_conv_kernel = getattr(self.cfg, 'rpn3d_conv_kernel', 3)

        if self.PlaneSweepVolume:
            self.build_cost = BuildCostVolume()

        self.anchor_angles = torch.as_tensor(self.cfg.ANCHOR_ANGLES)
        self.num_angles = self.cfg.num_angles

        self.feature_extraction = feature_extraction(cfg)

        res_dim = 64
        if self.PlaneSweepVolume:
            if not self.hg_firstconv:
                self.dres0 = nn.Sequential(convbn_3d(res_dim, res_dim, 3, 1, 1, gn=cfg.GN),
                                           nn.ReLU(inplace=True),
                                           convbn_3d(res_dim, res_dim, 3, 1, 1, gn=cfg.GN),
                                           nn.ReLU(inplace=True))

                self.dres1 = nn.Sequential(convbn_3d(res_dim, res_dim, 3, 1, 1, gn=cfg.GN),
                                           nn.ReLU(inplace=True),
                                           convbn_3d(res_dim, res_dim, 3, 1, 1, gn=cfg.GN))
            else:
                self.dres0 = hourglass(res_dim, gn=cfg.GN)

            self.hg_cv = self.cfg.hg_cv

            if self.hg_cv:
                self.dres2 = hourglass(res_dim, gn=cfg.GN)

            if self.loss_disp:
                self.classif1 = nn.Sequential(convbn_3d(res_dim, res_dim, 3, 1, 1, gn=cfg.GN),
                                              nn.ReLU(inplace=True),
                                              nn.Conv3d(res_dim, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.cat_disp = getattr(self.cfg, 'cat_disp', False)
        self.cat_img_feature = getattr(self.cfg, 'cat_img_feature', False)
        self.cat_right_img_feature = getattr(self.cfg, 'cat_right_img_feature', False)
        self.num_convs = getattr(self.cfg.RPN3D, 'NUM_CONVS', 4)
        self.num_3dconvs = getattr(self.cfg.RPN3D, 'NUM_3DCONVS', 1)
        assert self.num_3dconvs > 0

        RPN3D_INPUT_DIM = 0
        if self.PlaneSweepVolume: RPN3D_INPUT_DIM += res_dim
        if self.cat_disp: RPN3D_INPUT_DIM += 1
        if self.cat_img_feature: RPN3D_INPUT_DIM += self.cfg.RPN_CONVDIM
        if self.cat_right_img_feature: RPN3D_INPUT_DIM += self.cfg.RPN_CONVDIM

        if self.cfg.RPN3D_ENABLE:
            conv3d_dim = getattr(self.cfg, 'conv3d_dim', 64)

            self.rpn3d_conv = nn.Sequential(convbn_3d(RPN3D_INPUT_DIM, conv3d_dim, self.rpn3d_conv_kernel, 1, 
                1 if self.rpn3d_conv_kernel == 3 else 0, gn=cfg.GN), nn.ReLU(inplace=True))

            if self.num_3dconvs > 1:
                self.rpn_3dconv1 = nn.Sequential(convbn_3d(conv3d_dim, conv3d_dim, 3, 1, 1, gn=cfg.GN),
                    nn.ReLU(inplace=True))
            if self.num_3dconvs > 2:
                self.rpn_3dconv2 = nn.Sequential(convbn_3d(conv3d_dim, conv3d_dim, 3, 1, 1, gn=cfg.GN),
                    nn.ReLU(inplace=True))
            if self.num_3dconvs > 3:
                self.rpn_3dconv3 = nn.Sequential(convbn_3d(conv3d_dim, conv3d_dim, 3, 1, 1, gn=cfg.GN),
                    nn.ReLU(inplace=True))

            if self.hg_rpn_conv3d:
                self.hg_rpn3d_conv = hourglass(conv3d_dim, gn=cfg.GN)

            self.rpn3d_pool = torch.nn.AvgPool3d((1, 4, 1), stride=(1, 4, 1))
            self.rpn3d_conv2 = nn.Sequential(convbn(conv3d_dim * 5, conv3d_dim * 2, 3, 1, 1, 1, gn=cfg.GN),
                    nn.ReLU(inplace=True))

            if not self.hg_rpn_conv:
                self.rpn3d_conv3 = nn.Sequential(convbn(res_dim * 2, res_dim * 2, 3, 1, 1, 1, gn=cfg.GN),
                        nn.ReLU(inplace=True))
            else:
                self.rpn3d_conv3 = hourglass2d(res_dim * 2, gn=cfg.GN)

            self.rpn3d_cls_convs = nn.Sequential(convbn(res_dim * 2, res_dim * 2, 3, 1, 1, 1, gn=cfg.GN),
                    nn.ReLU(inplace=True))
            self.rpn3d_bbox_convs = nn.Sequential(convbn(res_dim * 2, res_dim * 2, 3, 1, 1, 1, gn=cfg.GN),
                    nn.ReLU(inplace=True))
            if self.num_convs > 1:
                self.rpn3d_cls_convs2 = nn.Sequential(convbn(res_dim * 2, res_dim * 2, 3, 1, 1, 1, gn=cfg.GN),
                        nn.ReLU(inplace=True))
                self.rpn3d_bbox_convs2 = nn.Sequential(convbn(res_dim * 2, res_dim * 2, 3, 1, 1, 1, gn=cfg.GN),
                        nn.ReLU(inplace=True))
            if self.num_convs > 2:
                self.rpn3d_cls_convs3 = nn.Sequential(convbn(res_dim * 2, res_dim * 2, 3, 1, 1, 1, gn=cfg.GN),
                        nn.ReLU(inplace=True))
                self.rpn3d_bbox_convs3 = nn.Sequential(convbn(res_dim * 2, res_dim * 2, 3, 1, 1, 1, gn=cfg.GN),
                        nn.ReLU(inplace=True))
            if self.num_convs > 3:
                self.rpn3d_cls_convs4 = nn.Sequential(convbn(res_dim * 2, res_dim * 2, 3, 1, 1, 1, gn=cfg.GN),
                        nn.ReLU(inplace=True))
                self.rpn3d_bbox_convs4 = nn.Sequential(convbn(res_dim * 2, res_dim * 2, 3, 1, 1, 1, gn=cfg.GN),
                        nn.ReLU(inplace=True))

            if self.class4angles:
                self.bbox_cls = nn.Conv2d(res_dim * 2, self.num_angles * self.num_classes, kernel_size=3, padding=1, stride=1)
            else:
                self.bbox_cls = nn.Conv2d(res_dim * 2, self.num_classes, kernel_size=3, padding=1, stride=1)

            centerness_dim = 1
            centerness_dim *= self.num_angles
            if self.centerness4class:
                centerness_dim *= self.num_classes
            self.bbox_centerness = nn.Conv2d(res_dim * 2, centerness_dim, kernel_size=3, padding=1, stride=1)

            self.each_angle_dim = 1

            self.hwl_dim = 3
            self.xyz_dim = 3
            # dx,dy,dz dh,dw,dl, [s,c, cls]xnum_angles
            self.bbox_reg = nn.Conv2d(res_dim * 2, self.num_classes * (self.xyz_dim + self.hwl_dim + self.num_angles * self.each_angle_dim), kernel_size=3, padding=1, stride=1)
            self.anchor_size = torch.as_tensor([cfg.RPN3D.ANCHORS_HEIGHT, cfg.RPN3D.ANCHORS_WIDTH, cfg.RPN3D.ANCHORS_LENGTH]).transpose(1, 0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        if self.cfg.RPN3D_ENABLE:
            torch.nn.init.normal_(self.bbox_cls.weight, std=0.1)
            torch.nn.init.constant_(self.bbox_cls.bias, 0)
            torch.nn.init.normal_(self.bbox_centerness.weight, std=0.1)
            torch.nn.init.constant_(self.bbox_centerness.bias, 0)
            torch.nn.init.normal_(self.bbox_reg.weight, std=0.02)
            torch.nn.init.constant_(self.bbox_reg.bias, 0)

            prior_prob = cfg.RPN3D.PRIOR_PROB
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.bbox_cls.bias, bias_value)

        default_baseline = 0.54
        default_fu = 721.5377
        default_scale = default_baseline * default_fu
        self.default_scale = default_scale

        affine_mat = torch.as_tensor([[[1., 0., 0.], [0., 1., 0.]]])
        affine_mat = affine_mat.repeat(self.maxdisp // self.downsample_disp, 1, 1)

        for i in range(self.maxdisp // self.downsample_disp):
            depth = ((i + 0.5) * self.downsample_disp + self.cfg.depth_min_intervals) * self.cfg.depth_interval
            affine_mat[self.maxdisp // self.downsample_disp - 1 - i, 0, 2] = default_scale / depth / self.downsample_disp
        self.affine_mat = affine_mat
        # depth: 2.0 -> 40.2 # interval 0.2m
        # disp : about 194.8 -> 9.69

        depth = torch.zeros((self.maxdisp))
        for i in range(self.maxdisp):
            depth[self.maxdisp - 1 - i] = (i+self.cfg.depth_min_intervals) * self.cfg.depth_interval
        self.depth = depth

        self.dispregression = disparityregression(self.maxdisp, cfg=self.cfg)

        self.CV_X_MIN, self.CV_Y_MIN, self.CV_Z_MIN = cfg.CV_X_MIN, cfg.CV_Y_MIN, cfg.CV_Z_MIN
        self.CV_X_MAX, self.CV_Y_MAX, self.CV_Z_MAX = cfg.CV_X_MAX, cfg.CV_Y_MAX, cfg.CV_Z_MAX
        self.X_MIN, self.Y_MIN, self.Z_MIN = cfg.X_MIN, cfg.Y_MIN, cfg.Z_MIN
        self.X_MAX, self.Y_MAX, self.Z_MAX = cfg.X_MAX, cfg.Y_MAX, cfg.Z_MAX
        self.VOXEL_X_SIZE, self.VOXEL_Y_SIZE, self.VOXEL_Z_SIZE = cfg.VOXEL_X_SIZE, cfg.VOXEL_Y_SIZE, cfg.VOXEL_Z_SIZE
        self.GRID_SIZE = cfg.GRID_SIZE

        zs = torch.arange(self.Z_MIN, self.Z_MAX, self.VOXEL_Z_SIZE) + self.VOXEL_Z_SIZE / 2.
        ys = torch.arange(self.Y_MIN, self.Y_MAX, self.VOXEL_Y_SIZE) + self.VOXEL_Y_SIZE / 2.
        xs = torch.arange(self.X_MIN, self.X_MAX, self.VOXEL_X_SIZE) + self.VOXEL_X_SIZE / 2.
        zs, ys, xs = torch.meshgrid(zs, ys, xs)
        coord_rect = torch.stack([xs, ys, zs], dim=-1)
        self.coord_rect = coord_rect

    def forward(self, left, right, calibs_fu, calibs_baseline, calibs_Proj, calibs_Proj_R=None):
        N = left.shape[0]

        refimg_fea, left_rpn_feature = self.feature_extraction(left)
        targetimg_fea, right_rpn_feature = self.feature_extraction(right)

        outputs = dict()

        if self.PlaneSweepVolume:
            affine_mat = self.affine_mat.cuda().clone().unsqueeze(0).repeat(N, 1, 1, 1)
            affine_mat[:, :, 0, 2] = affine_mat[:, :, 0, 2] * calibs_fu[:,None].cuda().float() * calibs_baseline[:,None].cuda().float() / self.default_scale
            cost = self.build_cost(refimg_fea, targetimg_fea, affine_mat[:,:,0,2])
            cost = cost.contiguous()

            if not self.hg_firstconv:
                cost0 = self.dres0(cost)
                cost0 = self.dres1(cost0) + cost0
            else:
                out0, pre0, post0 = self.dres0(cost, None, None)
                cost0 = out0

            if self.hg_cv:
                out1, pre1, post1 = self.dres2(cost0, None, None)
                out1 = out1 + cost0
                if self.loss_disp:
                    cost1 = self.classif1(out1)
                else:
                    cost1 = None

                out, cost = out1, cost1
            else:
                out0 = cost0
                if self.loss_disp:
                    cost0 = self.classif1(out0)
                else:
                    cost0 = None

                out, cost = out0, cost0
            
        outputs['depth_preds'] = []

        if self.PlaneSweepVolume and self.loss_disp:
            if self.hg_cv:
                cost1 = F.upsample(cost1, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear', align_corners=self.cfg.align_corners)
                cost1 = torch.squeeze(cost1, 1)
                pred1_softmax = F.softmax(cost1, dim=1)
                pred1 = self.dispregression(pred1_softmax, depth=self.depth.cuda())
                if self.training:
                    outputs['depth_preds'].append( pred1 )
                else:
                    outputs['depth_preds'] = pred1
            else:
                cost0 = F.upsample(cost0, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear', align_corners=self.cfg.align_corners)
                cost0 = torch.squeeze(cost0, 1)
                pred1_softmax = F.softmax(cost0, dim=1)
                pred1 = self.dispregression(pred1_softmax, depth=self.depth.cuda())
                if self.training:
                    outputs['depth_preds'].append( pred1 )
                else:
                    outputs['depth_preds'] = pred1

        if self.cfg.RPN3D_ENABLE:
            coord_rect = self.coord_rect.cuda()

            norm_coord_imgs = []
            for i in range(N):
                coord_img = torch.as_tensor(
                    project_rect_to_image(
                        coord_rect.reshape(-1, 3), 
                        calibs_Proj[i].float().cuda()
                    ).reshape(*self.coord_rect.shape[:3], 2), 
                dtype=torch.float32)

                coord_img = torch.cat([coord_img, self.coord_rect[..., 2:]], dim=-1)
                norm_coord_img = (coord_img - torch.as_tensor([self.CV_X_MIN, self.CV_Y_MIN, self.CV_Z_MIN])[None, None, None, :]) / \
                    (torch.as_tensor([self.CV_X_MAX, self.CV_Y_MAX, self.CV_Z_MAX]) - torch.as_tensor([self.CV_X_MIN, self.CV_Y_MIN, self.CV_Z_MIN]))[None, None, None, :]
                norm_coord_img = norm_coord_img * 2. - 1.
                norm_coord_imgs.append(norm_coord_img)
            norm_coord_imgs = torch.stack(norm_coord_imgs, dim=0)
            norm_coord_imgs = norm_coord_imgs.cuda()

            outputs['norm_coord_imgs'] = norm_coord_imgs
            outputs['coord_rect'] = coord_rect

            valids = (norm_coord_imgs[..., 0] >= -1.) & (norm_coord_imgs[..., 0] <= 1.) & \
                (norm_coord_imgs[..., 1] >= -1.) & (norm_coord_imgs[..., 1] <= 1.) & \
                (norm_coord_imgs[..., 2] >= -1.) & (norm_coord_imgs[..., 2] <= 1.)
            outputs['valids'] = valids
            valids = valids.float()

            if self.PlaneSweepVolume:
                # Retrieve Voxel Feature from Cost Volume Feature
                if self.cat_disp:
                    CV_feature = torch.cat([out, cost.detach()], dim= 1)
                else:
                    CV_feature = out

                Voxel = F.grid_sample(CV_feature, norm_coord_imgs)
                Voxel = Voxel * valids[:, None, :, :, :]

                if (self.voxel_attentionbydisp or (self.img_feature_attentionbydisp and self.cat_img_feature)):
                    pred_disp = F.grid_sample(pred1_softmax.detach()[:, None], norm_coord_imgs)
                    pred_disp = pred_disp * valids[:, None, :, :, :]

                if self.voxel_attentionbydisp:
                    Voxel = Voxel * pred_disp
            else:
                Voxel = None

            # Retrieve Voxel Feature from 2D Img Feature
            if self.cat_img_feature:
                RPN_feature = left_rpn_feature

                valids = (norm_coord_imgs[..., 0] >= -1.) & (norm_coord_imgs[..., 0] <= 1.) & \
                    (norm_coord_imgs[..., 1] >= -1.) & (norm_coord_imgs[..., 1] <= 1.)
                valids = valids.float() 

                Voxel_2D = []
                for i in range(N):
                    RPN_feature_per_im = RPN_feature[i:i+1]
                    for j in range(len(norm_coord_imgs[i])):
                        Voxel_2D_feature = F.grid_sample(RPN_feature_per_im, norm_coord_imgs[i, j:j+1, :, :, :2])
                        Voxel_2D.append(Voxel_2D_feature)
                Voxel_2D = torch.cat(Voxel_2D, dim=0)
                Voxel_2D = Voxel_2D.reshape(N, self.GRID_SIZE[0], -1, self.GRID_SIZE[1], self.GRID_SIZE[2]).transpose(1,2)
                Voxel_2D = Voxel_2D * valids[:, None, :, :, :]

                if self.img_feature_attentionbydisp:
                    Voxel_2D = Voxel_2D * pred_disp

                if Voxel is not None:
                    Voxel = torch.cat([Voxel, Voxel_2D], dim=1)
                else:
                    Voxel = Voxel_2D

            if self.cat_right_img_feature:
                RPN_feature = right_rpn_feature

                norm_coord_right_imgs = []
                for i in range(N):
                    coord_right_img = torch.as_tensor(
                        project_rect_to_image(
                            coord_rect.reshape(-1, 3), 
                            calibs_Proj_R[i].float().cuda()
                        ).reshape(*self.coord_rect.shape[:3], 2), 
                    dtype=torch.float32)

                    coord_right_img = torch.cat([coord_right_img, self.coord_rect[..., 2:]], dim=-1)
                    norm_coord_img = (coord_right_img - torch.as_tensor([self.CV_X_MIN, self.CV_Y_MIN, self.CV_Z_MIN])[None, None, None, :]) / \
                        (torch.as_tensor([self.CV_X_MAX, self.CV_Y_MAX, self.CV_Z_MAX]) - torch.as_tensor([self.CV_X_MIN, self.CV_Y_MIN, self.CV_Z_MIN]))[None, None, None, :]
                    norm_coord_img = norm_coord_img * 2. - 1.
                    norm_coord_right_imgs.append(norm_coord_img)
                norm_coord_right_imgs = torch.stack(norm_coord_right_imgs, dim=0)
                norm_coord_right_imgs = norm_coord_right_imgs.cuda()

                valids_R = (norm_coord_right_imgs[..., 0] >= -1.) & (norm_coord_right_imgs[..., 0] <= 1.) & \
                    (norm_coord_right_imgs[..., 1] >= -1.) & (norm_coord_right_imgs[..., 1] <= 1.) 
                valids_R = valids_R.float()

                Voxel_2D_R = []
                for i in range(N):
                    RPN_feature_per_im = RPN_feature[i:i+1]
                    for j in range(len(norm_coord_right_imgs[i])):
                        Voxel_2D_feature = F.grid_sample(RPN_feature_per_im, norm_coord_right_imgs[i, j:j+1, :, :, :2])
                        Voxel_2D_R.append(Voxel_2D_feature)
                Voxel_2D_R = torch.cat(Voxel_2D_R, dim=0)
                Voxel_2D_R = Voxel_2D_R.reshape(N, self.GRID_SIZE[0], -1, self.GRID_SIZE[1], self.GRID_SIZE[2]).transpose(1,2)
                Voxel_2D_R = Voxel_2D_R * valids_R[:, None, :, :, :]

                if self.img_feature_attentionbydisp:
                    Voxel_2D_R = Voxel_2D_R * pred_disp

                if Voxel is not None:
                    Voxel = torch.cat([Voxel, Voxel_2D_R], dim=1)
                else:
                    Voxel = Voxel_2D_R

            # (64, 190, 20, 300)
            Voxel = self.rpn3d_conv(Voxel) # (64, 190, 20, 300)

            if self.num_3dconvs > 1:
                Voxel = self.rpn_3dconv1(Voxel)
            if self.num_3dconvs > 2:
                Voxel = self.rpn_3dconv2(Voxel)
            if self.num_3dconvs > 3:
                Voxel = self.rpn_3dconv3(Voxel)

            if self.hg_rpn_conv3d:
                Voxel1, pre_Voxel, post_Voxel = self.hg_rpn3d_conv(Voxel, None, None)
                Voxel = Voxel1 + Voxel

            Voxel = self.rpn3d_pool(Voxel) # (64, 190, 5, 300)
            Voxel = Voxel.permute(0, 1, 3, 2, 4).reshape(N, -1, self.GRID_SIZE[0], self.GRID_SIZE[2]).contiguous()

            Voxel_BEV = self.rpn3d_conv2(Voxel)

            if not self.hg_rpn_conv:
                Voxel_BEV = self.rpn3d_conv3(Voxel_BEV)
            else:
                Voxel_BEV1, pre_BEV, post_BEV = self.rpn3d_conv3(Voxel_BEV, None, None)
                Voxel_BEV = Voxel_BEV1 # some bug

            Voxel_BEV_cls = self.rpn3d_cls_convs(Voxel_BEV)
            Voxel_BEV_bbox = self.rpn3d_bbox_convs(Voxel_BEV)
            if self.num_convs > 1:
                Voxel_BEV_cls = self.rpn3d_cls_convs2(Voxel_BEV_cls)
                Voxel_BEV_bbox = self.rpn3d_bbox_convs2(Voxel_BEV_bbox)
            if self.num_convs > 2:
                Voxel_BEV_cls = self.rpn3d_cls_convs3(Voxel_BEV_cls)
                Voxel_BEV_bbox = self.rpn3d_bbox_convs3(Voxel_BEV_bbox)
            if self.num_convs > 3:
                Voxel_BEV_cls = self.rpn3d_cls_convs4(Voxel_BEV_cls)
                Voxel_BEV_bbox = self.rpn3d_bbox_convs4(Voxel_BEV_bbox)

            bbox_cls = self.bbox_cls(Voxel_BEV_cls)
            if not self.fix_centerness_bug:
                bbox_reg = self.bbox_reg(Voxel_BEV_cls)
                bbox_centerness = self.bbox_centerness(Voxel_BEV_bbox)
            else:
                bbox_reg = self.bbox_reg(Voxel_BEV_bbox)
                bbox_centerness = self.bbox_centerness(Voxel_BEV_bbox)

            # dx, dy, h, w, l, q1, q2, q3, q4, dz
            N, C, H, W = bbox_reg.shape

            dxyz, dhwl, angle_reg = torch.split(bbox_reg.reshape(N, self.num_classes, C // self.num_classes, H, W), \
                    [self.xyz_dim, self.hwl_dim, self.each_angle_dim * self.num_angles], dim=2)

            # angle / orientation
            angle_reg = angle_reg.permute(0, 3, 4, 2, 1).reshape(-1, self.each_angle_dim * self.num_angles, self.num_classes)

            angle_range = np.pi * 2 / self.num_angles
            q = angle_reg.tanh() * angle_range / 2.
            q = q + self.anchor_angles.cuda()[None, :, None]
            sin_d, cos_d = torch.sin(q), torch.cos(q)

            # XYZ
            dxyz = dxyz[:, None, :].repeat(1, self.num_angles, 1, 1, 1, 1)

            dhwl = dhwl.permute(0, 3, 4, 1, 2).reshape(-1, self.num_classes, self.hwl_dim)
            dhwl = dhwl[:, None, :, :].repeat(1, self.num_angles, 1, 1)
            hwl = self.anchor_size.cuda().reshape(1, 1, self.num_classes, 3) * torch.exp(dhwl)
            hwl = hwl.reshape(-1, self.num_angles, self.num_classes, 3)

            if not self.box_corner_parameters:
                hwl = hwl.reshape(N, H, W, self.num_angles, self.num_classes, 3)
                hwl = hwl.permute(0, 3, 4, 5, 1, 2)

                q = q.reshape(N, H, W, self.num_angles, self.num_classes)
                q = q.permute(0, 3, 4, 1, 2)

                # N, num_angles, num_classes, dim, H, W
                bbox_reg = torch.cat( [dxyz, hwl, q[:, :, :, None]], dim=3)
                bbox_reg = bbox_reg.reshape(N, self.num_angles * self.num_classes * 7, H, W)
            else:
                box_corners = compute_corners_sc(
                    hwl.reshape(-1, 3), 
                    sin_d.reshape(-1), 
                    cos_d.reshape(-1)
                ).reshape(N, H, W, self.num_angles, self.num_classes, 3, 8)
                box_corners[:, :, :, :, :, 1, :] += hwl.reshape(N, H, W, self.num_angles, self.num_classes, 3)[:, :, :, :, :, 0:1] / 2.
                box_corners = box_corners.permute(0, 3, 4, 6, 5, 1, 2) 
                # (N, num_classes, num_angles, 8, 3, H, W)

                # (N, num_classes, num_angles, )
                bbox_reg = box_corners + dxyz[:, :, :, None]
                bbox_reg = bbox_reg.reshape(N, self.num_angles * self.num_classes * 24, H, W)

            outputs['bbox_cls'] = bbox_cls
            outputs['bbox_reg'] = bbox_reg
            outputs['bbox_centerness'] = bbox_centerness

        return outputs
