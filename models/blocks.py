import torch
import torch.nn as nn
from collections import OrderedDict

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(num_features=oup, eps=1e-05, momentum=0.1, affine=True),
        nn.ReLU(inplace=True)
    )

class MobileNetV2_Dilated(nn.Module):
    """docstring for MobileNetV2_Dilated"""
    def __init__(self):
        super(MobileNetV2_Dilated, self).__init__()

        self.modle = nn.Sequential(
            conv_bn(3, 32, 2),
            InvertedResidual(inp=32, oup=16, stride=1, dilate=1, expand_ratio=1),  # 1
            InvertedResidual(inp=16, oup=24, stride=2, dilate=1, expand_ratio=6),  # 2
            InvertedResidual(inp=24, oup=24, stride=1, dilate=1, expand_ratio=6),
            InvertedResidual(inp=24, oup=32, stride=2, dilate=1, expand_ratio=6),  # 3
            InvertedResidual(inp=32, oup=32, stride=1, dilate=1, expand_ratio=6),
            InvertedResidual(inp=32, oup=32, stride=1, dilate=1, expand_ratio=6),
            InvertedResidual(inp=32, oup=64, stride=1, dilate=2, expand_ratio=6),  # 4
            InvertedResidual(inp=64, oup=64, stride=1, dilate=2, expand_ratio=6),
            InvertedResidual(inp=64, oup=64, stride=1, dilate=2, expand_ratio=6),
            InvertedResidual(inp=64, oup=64, stride=1, dilate=2, expand_ratio=6),
            InvertedResidual(inp=64, oup=96, stride=1, dilate=4, expand_ratio=6),  # 5
            InvertedResidual(inp=96, oup=96, stride=1, dilate=4, expand_ratio=6),
            InvertedResidual(inp=96, oup=96, stride=1, dilate=4, expand_ratio=6),
            InvertedResidual(inp=96, oup=160, stride=1, dilate=8, expand_ratio=6),  # 6
            InvertedResidual(inp=160, oup=160, stride=1, dilate=8, expand_ratio=6),
            InvertedResidual(inp=160, oup=160, stride=1, dilate=8, expand_ratio=6),
            InvertedResidual(inp=160, oup=320, stride=1, dilate=16, expand_ratio=6), # 7
            )

    def forward(self, x):
        return self.modle(x)
        

# inverse-block in mobilenetv2 
class InvertedResidual(nn.Module):
    """docstring for InvertedResidual"""
    def __init__(self, inp, oup, stride, dilate, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.modle = nn.Sequential(
            #pw
            nn.Conv2d(in_channels=inp, out_channels=inp * expand_ratio,
                      kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(num_features=inp * expand_ratio, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU6(inplace=True),

            #dw
            nn.Conv2d(in_channels=inp * expand_ratio, out_channels=inp * expand_ratio,
                      kernel_size=3, stride=stride, padding=dilate, dilation=dilate,
                      groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(num_features=inp * expand_ratio, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU6(inplace=True),

            #pw
            nn.Conv2d(in_channels=inp * expand_ratio, out_channels=oup,
                      kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(num_features=oup, eps=1e-05, momentum=0.1, affine=True),
            )

    def forward(self, x):
        if(self.use_res_connect):
            return torch.add(x, 1, self.modle(x))
        else:
            return self.modle(x)
        
'''
class RFBlock(nn.Module):
    def __init__(self, in_chs, out_chs, scale=0.1, feat_res=(56, 112), aspp_sec=(12, 24, 36),
                 up_ratio=2, norm_act=InPlaceABN):
        super(RFBlock, self).__init__()
        self.scale = scale

        self.down_chs = nn.Sequential(OrderedDict([("norm_act", norm_act(in_chs)),
                                                   ("down_conv1x1", nn.Conv2d(in_chs, out_chs,
                                                                              kernel_size=1, stride=1,
                                                                              padding=0, bias=False))]))

        self.gave_pool = nn.Sequential(OrderedDict([("norm_act", norm_act(out_chs)),
                                                    ("gavg", nn.AdaptiveAvgPool2d((1, 1))),
                                                    ("conv1_0", nn.Conv2d(out_chs, out_chs,
                                                                          kernel_size=1, stride=1, padding=0,
                                                                          groups=1, bias=False, dilation=1)),
                                                    ("up0", nn.Upsample(size=feat_res, mode='bilinear'))]))

        self.branch0 = nn.Sequential(OrderedDict([("norm_act", norm_act(out_chs)),
                                                  ("conv1x1", nn.Conv2d(out_chs, out_chs,
                                                                        kernel_size=1, stride=1,
                                                                        padding=0, bias=False)),
                                                  ("norm_act", norm_act(out_chs)),
                                                  ("aconv1", nn.Conv2d(out_chs, out_chs,
                                                                       kernel_size=3, stride=1,
                                                                       padding=1, dilation=1,
                                                                       bias=False))]))

        self.branch1 = nn.Sequential(OrderedDict([("norm_act", norm_act(out_chs)),
                                                  ("conv1x3", nn.Conv2d(out_chs, (out_chs // 2) * 3,
                                                                        kernel_size=(1, 3), stride=1,
                                                                        padding=(0, 1), bias=False)),
                                                  ("norm_act", norm_act((out_chs // 2) * 3)),
                                                  ("conv3x1", nn.Conv2d((out_chs // 2) * 3, out_chs,
                                                                        kernel_size=(3, 1), stride=1,
                                                                        padding=(1, 0), bias=False)),
                                                  ("norm_act", norm_act(out_chs)),
                                                  ("aconv3", nn.Conv2d(out_chs, out_chs,
                                                                       kernel_size=3, stride=1,
                                                                       padding=aspp_sec[0],
                                                                       dilation=aspp_sec[0],
                                                                       bias=False))]))

        self.branch2 = nn.Sequential(OrderedDict([("norm_act", norm_act(out_chs)),
                                                  ("conv1x5", nn.Conv2d(out_chs, (out_chs // 2) * 3,
                                                                        kernel_size=(1, 5), stride=1,
                                                                        padding=(0, 2), bias=False)),
                                                  ("norm_act", norm_act((out_chs // 2) * 3)),
                                                  ("conv5x1", nn.Conv2d((out_chs // 2) * 3, out_chs,
                                                                        kernel_size=(5, 1), stride=1,
                                                                        padding=(2, 0), bias=False)),
                                                  ("norm_act", norm_act(out_chs)),
                                                  ("aconv5", nn.Conv2d(out_chs, out_chs,
                                                                       kernel_size=3, stride=1,
                                                                       padding=aspp_sec[1],
                                                                       dilation=aspp_sec[1],
                                                                       bias=False))]))

        self.branch3 = nn.Sequential(OrderedDict([("norm_act", norm_act(out_chs)),
                                                  ("conv1x7", nn.Conv2d(out_chs, (out_chs // 2) * 3,
                                                                        kernel_size=(1, 7), stride=1,
                                                                        padding=(0, 3), bias=False)),
                                                  ("norm_act", norm_act((out_chs // 2) * 3)),
                                                  ("conv7x1", nn.Conv2d((out_chs // 2) * 3, out_chs,
                                                                        kernel_size=(7, 1), stride=1,
                                                                        padding=(3, 0), bias=False)),
                                                  ("norm_act", norm_act(out_chs)),
                                                  ("aconv7", nn.Conv2d(out_chs, out_chs,
                                                                       kernel_size=3, stride=1,
                                                                       padding=aspp_sec[2],
                                                                       dilation=aspp_sec[2],
                                                                       bias=False))]))

        self.conv_linear = nn.Sequential(OrderedDict([("conv1x1_linear", nn.Conv2d(out_chs * 5, out_chs,
                                                                                   kernel_size=1, stride=1,
                                                                                   padding=0, bias=False))]))

        self.upsampling = nn.Upsample(size=(int(feat_res[0] * up_ratio),
                                            int(feat_res[1] * up_ratio)),
                                      mode='bilinear')

    def forward(self, x):
        down = self.down_chs(x)
        out = torch.cat([self.gave_pool(down.clone()),
                         self.branch0(down.clone()),
                         self.branch1(down.clone()),
                         self.branch2(down.clone()),
                         self.branch3(down.clone())], dim=1)

        return self.upsampling(torch.add(self.conv_linear(out), self.scale, down))  # out=input+value×other
'''


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fcs = nn.Sequential(nn.Linear(channel, int(channel/reduction)),
                                 nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                 nn.Linear(int(channel/reduction), channel),
                                 nn.Sigmoid())

    def forward(self, x):
        bahs, chs, _, _ = x.size()

        # Returns a new tensor with the same data as the self tensor but of a different size.
        y = self.avg_pool(x).view(bahs, chs)
        y = self.fcs(y).view(bahs, chs, 1, 1)
        return torch.mul(x, y)

# no downsamplng
class SCSEBlock(nn.Module):
    """docstring for SCSEBlock"""
    def __init__(self, channel, reduction=16):
        super(SCSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(channel//reduction)),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(channel//reduction), channel),
                                                nn.Sigmoid())

        self.spatial_se = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=1,
                                                  stride=1, padding=0, bias=False),
                                        nn.Sigmoid())

    def forward(self, x):
        bahs, chs, _, _ = x.size()

        # Returns a new tensor with the same data as the self tensor but of a different size.
        chn_se = self.avg_pool(x).view(bahs, chs)
        chn_se = self.channel_excitation(chn_se).view(bahs, chs, 1, 1)
        chn_se = torch.mul(x, chn_se)

        spa_se = self.spatial_se(x)
        spa_se = torch.mul(x, spa_se)
        return torch.add(chn_se, 1, spa_se)


class ASPPBlock(nn.Module):
    """docstring for ASPPBlock"""
    def __init__(self, inp, oup):
        super(ASPPBlock, self).__init__()
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False, groups=1, dilation=1),
            nn.UpsamplingBilinear2d((60, 80)), ### how to resuze to origin size
            nn.BatchNorm2d(oup)
            )

        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False, groups=1, dilation=1),
            nn.BatchNorm2d(oup)
            )

        self.aspp_1 = nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size=3, stride=1, padding=6, bias=False, groups=1, dilation=6),
            nn.BatchNorm2d(oup)
            )

        self.aspp_2 = nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size=3, stride=1, padding=12, bias=False, groups=1, dilation=12),
            nn.BatchNorm2d(oup)
            )

        self.aspp_3 = nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size=3, stride=1, padding=18, bias=False, groups=1, dilation=18),
            nn.BatchNorm2d(oup)
            )

        self.last_conv = nn.Sequential(
            nn.Conv2d(5*oup, oup, kernel_size=1, stride=1, padding=0, bias=False, groups=1, dilation=1),
            nn.BatchNorm2d(oup),
            nn.Dropout2d(p=0.2, inplace=True)
            )
        
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, segSize=None):

        x1 = self.image_pool(x)
        x2 = self.conv_1x1(x)
        x3 = self.aspp_1(x)
        x4 = self.aspp_2(x)
        x5 = self.aspp_3(x)

        # print("###########################")
        # print(x.size())
        # print(x1.size())
        # print(x2.size())
        # print(x3.size())
        # print(x4.size())
        # print(x5.size())
        # print("###########################")

        out = torch.cat(
            [x1, x2, x3, x4, x5], dim=1
            )

        out = self.last_conv(out)
        return self.upsampling(out)
