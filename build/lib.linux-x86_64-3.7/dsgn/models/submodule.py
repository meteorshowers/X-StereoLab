from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
from torch.nn import BatchNorm2d

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation, gn=False, groups=32):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes) if not gn else nn.GroupNorm(groups, out_planes))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad, gn=False, groups=32):
    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=False),
                         nn.BatchNorm3d(out_planes) if not gn else nn.GroupNorm(groups, out_planes))

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation, gn=False):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation, gn=gn),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation, gn=gn)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out

class disparityregression(nn.Module):
    def __init__(self, maxdisp, cfg):
        super(disparityregression, self).__init__()
        self.disp = Variable(torch.Tensor(np.array(range(maxdisp))).cuda(), requires_grad=False)

    def forward(self, x, depth):
        out = torch.sum(x * depth[None, :, None, None],1)
        return out

class hourglass(nn.Module):
    def __init__(self, inplanes, gn=False):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1, gn=gn),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1, gn=gn)

        self.conv3 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=2, pad=1, gn=gn),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1, gn=gn),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes * 2) if not gn else nn.GroupNorm(32, inplanes * 2))  # +conv2

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes) if not gn else nn.GroupNorm(32, inplanes))  # +x

    def forward(self, x, presqu, postsqu):

        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16

        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu, inplace=True)  # in:1/16 out:1/8
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        out = self.conv6(post)  # in:1/8 out:1/4

        return out, pre, post

class hourglass2d(nn.Module):
    def __init__(self, inplanes, gn=False):
        super(hourglass2d, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1, dilation=1, gn=gn),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1, dilation=1, gn=gn)

        self.conv3 = nn.Sequential(convbn(inplanes * 2, inplanes * 2, kernel_size=3, stride=2, pad=1, dilation=1, gn=gn),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1, dilation=1, gn=gn),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(inplanes * 2, inplanes * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm2d(inplanes * 2) if not gn else nn.GroupNorm(32, inplanes * 2))  # +conv2

        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(inplanes * 2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm2d(inplanes) if not gn else nn.GroupNorm(32, inplanes))  # +x

    def forward(self, x, presqu, postsqu):

        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16

        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu, inplace=True)  # in:1/16 out:1/8
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        out = self.conv6(post)  # in:1/8 out:1/4

        return out, pre, post

class feature_extraction(nn.Module):
    def __init__(self, cfg):
        super(feature_extraction, self).__init__()

        self.cfg = cfg
        self.RPN3D_ENABLE = self.cfg.RPN3D_ENABLE
        self.cat_img_feature = getattr(self.cfg, 'cat_img_feature', False)
        self.rpn_onemore_conv = getattr(self.cfg, 'RPN_ONEMORE_CONV', False)
        self.rpn_onemore_dim = getattr(self.cfg, 'RPN_ONEMORE_DIM', 256)
        self.img_feature_relu = getattr(self.cfg, 'img_feature_relu', True)
        self.branch = getattr(self.cfg, 'branch', True)

        self.backbone = getattr(self.cfg, 'backbone', 'reslike-det-small')
        if self.backbone == 'reslike-det':
            first_dim = 64
            dims = [64, 128, 192, 256]
            nr_convs = [3, 6, 12, 4]
            branch_dim = 32
            lastconv_dim = [256, 32]
        elif self.backbone == 'reslike-det-small':
            first_dim = 64
            dims = [32, 64, 128, 192]
            nr_convs = [3, 6, 12, 4]
            branch_dim = 32
            lastconv_dim = [256, 32]
        elif self.backbone == 'reslike-det-small-fixfirst':
            first_dim = 16
            dims = [32, 64, 128, 192]
            nr_convs = [3, 6, 12, 4]
            branch_dim = 32
            lastconv_dim = [256, 32]
        elif self.backbone == 'reslike50-det-small-fixfirst':
            first_dim = 16
            dims = [32, 64, 128, 256]
            nr_convs = [3, 4, 6, 3]
            branch_dim = 32
            lastconv_dim = [256, 32]
        elif self.backbone == 'reslike50-det-tiny':
            first_dim = 8
            dims = [16, 32, 64, 128]
            nr_convs = [3, 4, 6, 3]
            branch_dim = 32
            lastconv_dim = [128, 32]
        else:
            raise ValueError('Invalid backbone {}.'.format(self.backbone))

        self.inplanes = first_dim

        self.firstconv = nn.Sequential(convbn(3, first_dim, 3, 2, 1, 1, gn=cfg.GN if first_dim >= 32 else False),
                                       nn.ReLU(inplace=True),
                                       convbn(first_dim, first_dim, 3, 1, 1, 1, gn=cfg.GN if first_dim >= 32 else False),
                                       nn.ReLU(inplace=True),
                                       convbn(first_dim, first_dim, 3, 1, 1, 1, gn=cfg.GN if first_dim >= 32 else False),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, dims[0], nr_convs[0], 1,1,1, gn=cfg.GN if dims[0] >= 32 else False)
        self.layer2 = self._make_layer(BasicBlock, dims[1], nr_convs[1], 2,1,1, gn=cfg.GN) 
        self.layer3 = self._make_layer(BasicBlock, dims[2], nr_convs[2], 1,1,1, gn=cfg.GN)
        self.layer4 = self._make_layer(BasicBlock, dims[3], nr_convs[3], 1,1,2, gn=cfg.GN)

        if self.branch:
            self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64,64)),
                                         convbn(dims[3], branch_dim, 1, 1, 0, 1, gn=cfg.GN, groups=min(32, branch_dim)),
                                         nn.ReLU(inplace=True))

            self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32,32)),
                                         convbn(dims[3], branch_dim, 1, 1, 0, 1, gn=cfg.GN, groups=min(32, branch_dim)),
                                         nn.ReLU(inplace=True))

            self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16,16)),
                                         convbn(dims[3], branch_dim, 1, 1, 0, 1, gn=cfg.GN, groups=min(32, branch_dim)),
                                         nn.ReLU(inplace=True))

            self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8,8)),
                                         convbn(dims[3], branch_dim, 1, 1, 0, 1, gn=cfg.GN, groups=min(32, branch_dim)),
                                         nn.ReLU(inplace=True))

        if self.branch:
            concat_dim = branch_dim * 4 + dims[1] + dims[3] + dims[2]
        else:
            concat_dim = dims[1] + dims[3] + dims[2]

        self.PlaneSweepVolume = getattr(cfg, 'PlaneSweepVolume', True)
        if self.PlaneSweepVolume:
            self.lastconv = nn.Sequential(convbn(concat_dim, lastconv_dim[0], 3, 1, 1, 1, gn=cfg.GN),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(lastconv_dim[0], lastconv_dim[1], kernel_size=1, padding=0, stride = 1, bias=False))

        if self.cfg.RPN3D_ENABLE and self.cat_img_feature:
            if self.rpn_onemore_conv:
                rpnconvs = [convbn(concat_dim, self.rpn_onemore_dim, 3, 1, 1, 1, gn=cfg.GN),
                                          nn.ReLU(inplace=True),
                                          convbn(self.rpn_onemore_dim, self.cfg.RPN_CONVDIM, 3, 1, 1, 1, gn=cfg.GN, groups=(32 if self.cfg.RPN_CONVDIM % 32 == 0 else 16))]
            else:
                rpnconvs = [convbn(concat_dim, self.cfg.RPN_CONVDIM, 3, 1, 1, 1, gn=cfg.GN, groups=(32 if self.cfg.RPN_CONVDIM % 32 == 0 else 16))]
            if self.img_feature_relu:
                rpnconvs.append( nn.ReLU(inplace=True) )
            self.rpnconv = nn.Sequential( *rpnconvs )

    def _make_layer(self, block, planes, blocks, stride, pad, dilation, gn=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion) if not gn else nn.GroupNorm(32, planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation, gn=gn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pad,dilation, gn=gn))

        return nn.Sequential(*layers)

    def forward(self, x):
        output      = self.firstconv(x)         ; #print('conv1', output.shape)           # (1, 32, 192, 624)
        output      = self.layer1(output)       ; #print('conv2', output.shape)           # (1, 32, 192, 624)
        output_raw  = self.layer2(output)       ; #print('conv3', output_raw.shape)       # (1, 64, 96, 312)
        output_mid  = self.layer3(output_raw)   ; #print('conv4', output.shape)           # (1, 128, 96, 312)
        output_skip = self.layer4(output_mid)   ; #print('conv5', output_skip.shape)      # (1, 128, 96, 312)

        if self.branch:
            output_branch1 = self.branch1(output_skip) ; #print('b1', output_branch1.shape) # (1, 32, 1, 4) # avgpool 64
            output_branch1 = F.interpolate(output_branch1, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear', align_corners=self.cfg.align_corners) # (1, 32, 96, 312)

            output_branch2 = self.branch2(output_skip) ; #print('b2', output_branch2.shape)# (1, 32, 3, 9)
            output_branch2 = F.interpolate(output_branch2, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear', align_corners=self.cfg.align_corners)

            output_branch3 = self.branch3(output_skip) ; #print('b3', output_branch3.shape)# (1, 32, 6, 19)
            output_branch3 = F.interpolate(output_branch3, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear', align_corners=self.cfg.align_corners)

            output_branch4 = self.branch4(output_skip) ; #print('b4', output_branch4.shape)# (1, 32, 12, 39)
            output_branch4 = F.interpolate(output_branch4, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear', align_corners=self.cfg.align_corners)

        if self.branch:
            concat_feature = torch.cat((output_raw, output_mid, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1) ; #print('cat', concat_feature.shape)
        else:
            concat_feature = torch.cat((output_raw, output_mid, output_skip), 1)
        
        if self.RPN3D_ENABLE and self.cat_img_feature:
            rpn_feature = self.rpnconv(concat_feature)
        else:
            rpn_feature = None

        if self.PlaneSweepVolume:
            output_feature = self.lastconv(concat_feature) ; #print('last', output_feature.shape)
        else:
            output_feature = None

        return output_feature, rpn_feature
