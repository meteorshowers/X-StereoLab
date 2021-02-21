import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.backends.cudnn as cudnn

def convbn(in_channel, out_channel, kernel_size, stride, pad, dilation):
    
    return nn.Sequential(
        nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation if dilation>1 else pad,
            dilation=dilation),
       nn.BatchNorm2d(out_channel))

def convbn_3d(in_channel, out_channel, kernel_size, stride, pad):

    return nn.Sequential(
        nn.Conv3d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            padding=pad,
            stride=stride),
       nn.BatchNorm3d(out_channel))

class ConvolutionBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride, downsample, pad, dilation):
        super(ConvolutionBlock, self).__init__()
        self.conv1 = nn.Sequential(
            convbn(in_channel, out_channel, 3, stride, pad, dilation),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        # out = x + out
        return out

class ResNetBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride, downsample, pad, dilation):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = x + out
        return out

class Siamese_Tower(nn.Module):
    def __init__(self):
        super(Siamese_Tower, self).__init__()

        self.conv_begin = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)

        self.residual_blocks = nn.ModuleList()
        for _ in range(3):
            self.residual_blocks.append(
                ResNetBlock(
                    32, 32, stride=1, downsample=None, pad=1, dilation=1))


        self.downsample = nn.ModuleList()
        for _ in range(3):
            self.downsample.append(
                ConvolutionBlock(
                    32, 32, stride=2, downsample=None, pad=1, dilation=1))

        self.conv_end = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

    def forward(self, rgb_img):
        output = rgb_img
        output = self.conv_begin(output)

        for block in self.residual_blocks:
            output = block(output)

        for block in self.downsample:
            output = block(output)
        
        output = self.conv_end (output)

        return output

class Disparity_Refinement(nn.Module):
    #return: full_res disparity
    def __init__(self, in_channel):
        super(Disparity_Refinement, self).__init__()


        self.conv2d_feature_img = nn.Sequential(
            convbn(in_channel, 16, kernel_size=3, stride=1, pad=1, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.residual_astrous_blocks_img = nn.ModuleList()
        astrous_list = [1, 2]
        for di in astrous_list:
            self.residual_astrous_blocks_img.append(
                ResNetBlock(
                    16, 16, stride=1, downsample=None, pad=1, dilation=di))

        self.conv2d_feature_disp = nn.Sequential(
            convbn(in_channel, 16, kernel_size=3, stride=1, pad=1, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.residual_astrous_blocks_disp = nn.ModuleList()
        astrous_list = [1, 2]
        for di in astrous_list:
            self.residual_astrous_blocks_disp.append(
                ResNetBlock(
                    16, 16, stride=1, downsample=None, pad=1, dilation=di))

        self.residual_astrous_blocks_cated = nn.ModuleList()
        astrous_list = [4, 8, 1, 1]
        for di in astrous_list:
            self.residual_astrous_blocks_cated.append(
                ResNetBlock(
                    32, 32, stride=1, downsample=None, pad=1, dilation=di))

        self.conv_end = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, low_disparity, corresponding_rgb):

        feature_disp = self.conv2d_feature_disp(low_disparity)
        feature_img = self.conv2d_feature_img(corresponding_rgb)
        feature_cated = torch.cat([feature_disp, feature_img], dim=1)

        Disparity_Residual = self.conv_end(feature_cated)

        
        return Disparity_Residual + low_disparity 

class Invalidation_Net(nn.Module):
    #return: full_res Invalidation
    def __init__(self):
        super(Invalidation_Net, self).__init__()
        
        self.residual_blocks1 = nn.ModuleList()
        for _ in range(5):
            self.residual_blocks1.append(
                ResNetBlock(
                    64, 64, stride=1, downsample=None, pad=1, dilation=1))
        self.conv_end1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        self.conv_begin = ConvolutionBlock(
                    3, 32, stride=1, downsample=None, pad=1, dilation=1)

        self.residual_blocks2 = nn.ModuleList()
        for _ in range(4):
            self.residual_blocks2.append(
                ResNetBlock(
                    32, 32, stride=1, downsample=None, pad=1, dilation=1))
        
        self.conv_end2 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, left_tower, right_tower, input_img, full_res_disparity):

        output = torch.cat([left_tower, right_tower], dim=1)

        for block in self.residual_blocks1:
            
            output = block(output)
        
        Low_Res_invalidation_small = self.conv_end1(output)
        
        pred = Low_Res_invalidation_small * input_img.size()[-1] / Low_Res_invalidation_small.size()[-1]

        Low_Res_invalidation = F.upsample(
                pred,
                size=input_img.size()[-2:],
                mode='bilinear',
                align_corners=False)


        output = torch.cat([input_img, Low_Res_invalidation,full_res_disparity], dim=1)
        output = self.conv_begin(output)
        for block in self.residual_blocks2:
            output = block(output)
        Invalidation_Residual = self.conv_end2(output)

        return Invalidation_Residual + Low_Res_invalidation

class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        self.disp = torch.FloatTensor(
            np.reshape(np.array(range(maxdisp)), [1, maxdisp, 1, 1])).cuda()

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        out = torch.sum(x * disp, 1)
        return out


class Active_StereoNet(nn.Module):
    def __init__(self, maxdisp=144):
        super(Active_StereoNet, self).__init__()
        self.maxdisp = maxdisp
        self.Siamese_Tower = Siamese_Tower()
        self.filter = nn.ModuleList()
        for _ in range(4):
            self.filter.append(
                nn.Sequential(
                    convbn_3d(32, 32, kernel_size=3, stride=1, pad=1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        self.conv3d_alone = nn.Conv3d(
            32, 1, kernel_size=3, stride=1, padding=1)
        
        self.Disparity_Refinement = Disparity_Refinement(in_channel=1)
        self.Invalidation_Net = Invalidation_Net()

    
    def forward(self, left, right):
        disp = (self.maxdisp + 1) // 8
        refimg_feature = self.Siamese_Tower(left)
        targetimg_feature = self.Siamese_Tower(right)
        
        def calculate(refimg_feature, targetimg_feature, img, type):

            # matching
            cost = torch.FloatTensor(refimg_feature.size()[0],
                                    refimg_feature.size()[1],
                                    disp,
                                    refimg_feature.size()[2],
                                    refimg_feature.size()[3]).zero_().cuda()

            if type == 'left':
                for i in range(disp):
                    if i > 0:
                        cost[:, :, i, :, i:] = refimg_feature[ :, :, :, i:] - targetimg_feature[:, :, :, :-i]
                    else: 
                        cost[:, :, i, :, :] = refimg_feature - targetimg_feature

            if type == 'right':
                for i in range(disp):
                    if i > 0:
                        cost[:, :, i, :, :-i] = refimg_feature[ :, :, :, :-i] - targetimg_feature[:, :, :, i:]
                    else: 
                        cost[:, :, i, :, :] = refimg_feature - targetimg_feature
            cost = cost.contiguous()

            for f in self.filter:
                cost = f(cost)
            cost = self.conv3d_alone(cost)
            cost = torch.squeeze(cost, 1)
            pred = F.softmax(cost, dim=1)
            pred = disparityregression(disp)(pred)

            pred = pred * img.size()[-1] / pred.size()[-1]

            res_disparity = F.upsample(
                    torch.unsqueeze(pred, dim=1),
                    size=img.size()[-2:],
                    mode='bilinear',
                    align_corners=False)

            return res_disparity

        res_disparityL = calculate(refimg_feature, targetimg_feature, left, 'left')
        res_disparityR = calculate(targetimg_feature, refimg_feature, right, 'right')

        Full_res_disparityL = self.Disparity_Refinement(res_disparityL, left)
        Full_res_disparityR = self.Disparity_Refinement(res_disparityR, right)
        
        
        # Full_res_invalidation = self.Invalidation_Net(refimg_feature, targetimg_feature, left, Full_res_disparityL)



        return Full_res_disparityL, Full_res_disparityR,  Full_res_disparityR
        # return Full_res_disparityL, Full_res_disparityL,Full_res_disparityL



if __name__ == '__main__':
    model = Active_StereoNet().cuda()
    # model.eval()
    import time
    import datetime
    import torch
    # torch.backends.cudnn.benchmark = True    
    input = torch.FloatTensor(1,1,720,1280).zero_().cuda()
    with torch.no_grad():

        from thop import clever_format
        from thop import profile


        flops, params = profile(model, inputs=(input, input))
        flops, params = clever_format([flops, params], "%.3f")
        print(flops, params)

    





    


