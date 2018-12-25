import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torch.nn.functional as F
from . import resnet, resnext, mobilenet
from lib.nn import SynchronizedBatchNorm2d
from lib.utils import as_numpy, mark_volatile
from utils import AverageMeter, colorEncode, accuracy, intersectionAndUnion
from scipy.io import loadmat
from torchvision.models import resnet18

import cv2
import os
import math
from .blocks import ASPPBlock, SCSEBlock, InvertedResidual, conv_bn
from collections import OrderedDict
from functools import partial

from models.build_contextpath import build_contextpath
from .AlignedXception import AlignedXception
from .xception import *


import random
def visualize_result(preds):
    colors = loadmat('data/color150.mat')['colors']

    # prediction
    pred_color = colorEncode(preds, colors)
    pred_color = cv2.resize(pred_color, (480, 640))

    count = str(random.randint(0,1000))
    img_name = count+'.jpg'
    cv2.imwrite(os.path.join('temp/result_during_training',
                img_name.replace('.jpg', '.png')), pred_color)



class SegmentationModuleBase(nn.Module):
    def __init__(self):
        super(SegmentationModuleBase, self).__init__()

    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc


class SegmentationModule(SegmentationModuleBase):
    def __init__(self, net_enc, net_dec, crit, deep_sup_scale=None):
        super(SegmentationModule, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.crit = crit
        self.deep_sup_scale = deep_sup_scale

        self.isMacnet = True

    def forward(self, feed_dict, *, segSize=None):
        if segSize is None: # training
            if self.isMacnet:
                if self.deep_sup_scale is not None:
                    (pred, pred_deepsup) = self.encoder(feed_dict['img_data'])
                else:
                    pred =  self.encoder(feed_dict['img_data'])
                    ## visualize_result here ###
                    # pred_ = pred[0];
                    # pred_temp = torch.zeros(1, 150, 60, 80) # 1/8
                    # pred_temp = pred_temp + pred_.cpu()
                    # _, preds = torch.max(pred_temp.data.cpu(), dim=1)
                    # preds = as_numpy(preds.squeeze(0))
                    # visualize_result(preds)
            else:
                if self.deep_sup_scale is not None: # use deep supervision technique
                    (pred, pred_deepsup) = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True))
                else:
                    pred = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True))

            loss = self.crit(pred, feed_dict['seg_label'])
            if self.deep_sup_scale is not None:
                loss_deepsup = self.crit(pred_deepsup, feed_dict['seg_label'])
                loss = loss + loss_deepsup * self.deep_sup_scale

                '''
                ## add edge info ###
                edge = feed_dict['edge_data']
                # import pdb
                # pdb.set_trace()
                edge[edge==255]=0.4
                edge[edge==0]=0.6
                loss = torch.mul(loss, edge)
                '''

            acc = self.pixel_acc(pred, feed_dict['seg_label'])
            return loss, acc
        else: # inference
            if self.isMacnet:
                pred =  self.encoder(feed_dict['img_data'], segSize=segSize)
            else:
                pred = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True), segSize=segSize)
            return pred


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            SynchronizedBatchNorm2d(out_planes),
            # nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )


class ModelBuilder():
    # custom weights initialization
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
        #elif classname.find('Linear') != -1:
        #    m.weight.data.normal_(0.0, 0.0001)

    def build_encoder(self, arch='BiseNet', fc_dim=512, weights='', use_softmax=False):
        # print("weights=",weights)
        pretrained = True if len(weights) == 0 else False
        # print("pretrained=", pretrained)
        # print("arch=", arch)
        if arch == 'resnet34':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet34_dilated8':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=8)
        elif arch == 'resnet34_dilated16':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=16)
        elif arch == 'resnet50':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet50_dilated8':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=8)
        elif arch == 'resnet50_dilated16':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=16)
        elif arch == 'resnet101':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet101_dilated8':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=8)
        elif arch == 'resnet101_dilated16':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=16)
        elif arch == 'resnext101':
            orig_resnext = resnext.__dict__['resnext101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnext) # we can still use class Resnet
        ###########  persional testing  ########
        elif arch == 'mobilenetv1':
            orig_mobilenet = mobilenet.__dict__['mobilenetv1'](False)
            net_encoder = Mobilenet(orig_mobilenet)
        elif arch == 'mobilenetv2':
            orig_mobilenet = mobilenet.__dict__['mobilenetv2']()
            net_encoder = MobilenetV2(orig_mobilenet)
        elif arch == 'resnet_esp':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = Resnet_esp(orig_resnet)
        elif arch == 'Macnet':
            orig_mobilenet = mobilenet.__dict__['mobilenetv2']()
            net_encoder = Macnet(orig_mobilenet, use_softmax)
        elif arch == 'MacnetV2':
            net_encoder = MacnetV2(use_softmax=use_softmax)
        elif arch == 'BiseNet':
            net_encoder = BiSeNet(use_softmax=use_softmax)
        ###########  end for testing   ########
        else:
            raise Exception('Architecture undefined!')

        # net_encoder.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_encoder')
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_encoder

    def build_decoder(self, arch='ppm_bilinear_deepsup',
                      fc_dim=512, num_class=150,
                      weights='', use_softmax=False):
        if arch == 'c1_bilinear_deepsup':
            net_decoder = C1BilinearDeepSup(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'c1_bilinear':
            net_decoder = C1Bilinear(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'ppm_bilinear':
            net_decoder = PPMBilinear(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'ppm_bilinear_deepsup':
            net_decoder = PPMBilinearDeepsup(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'upernet_lite':
            net_decoder = UPerNet(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=256)
        elif arch == 'upernet':
            net_decoder = UPerNet(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=512)
        elif arch == 'upernet_tmp':
            net_decoder = UPerNetTmp(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=512)
        else:
            raise Exception('Architecture undefined!')

        net_decoder.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_decoder')
            net_decoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_decoder


###########BiSeNet#################BiSeNet#####BiSeNet#########BiSeNet#############BiSeNet##########


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2,padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        # self.bn = SynchronizedBatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))

class Spatial_path(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convblock1 = ConvBlock(in_channels=3, out_channels=64)
        self.convblock2 = ConvBlock(in_channels=64, out_channels=128)
        self.convblock3 = ConvBlock(in_channels=128, out_channels=256)

    def forward(self, input):
        x = self.convblock1(input)
        x = self.convblock2(x)
        x = self.convblock3(x)
        return x


class AttentionRefinementModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        # self.bn = SynchronizedBatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels
    def forward(self, input):
        # global average pooling
        x = torch.mean(input, 3, keepdim=True)
        x = torch.mean(x, 2, keepdim=True)
        assert self.in_channels == x.size(1), 'in_channels and out_channels should all be {}'.format(x.size(1))
        x = self.conv(x)
        # x = self.sigmoid(self.bn(x))
        x = self.sigmoid(x)
        # channels of input and x should be same
        x = torch.mul(input, x)
        return x


class FeatureFusionModule(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # self.in_channels = input_1.channels + input_2.channels
        # self.in_channels = 3328
        self.in_channels = 1024
        self.convblock = ConvBlock(in_channels=self.in_channels, out_channels=num_classes, stride=1)
        self.conv1 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_1, input_2):
        # print("input_1 channels=", input_1.size())
        # print("input_2 channels=", input_2.size())
        x = torch.cat((input_1, input_2), dim=1)
        assert self.in_channels == x.size(1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        feature = self.convblock(x)
        x = torch.mean(feature, 3, keepdim=True)
        x = torch.mean(x, 2 ,keepdim=True)
        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv1(x))
        x = torch.mul(feature, x)
        x = torch.add(x, feature)
        return x


class PSPModule(torch.nn.Module):
    """docstring for PSPModule"""
    def __init__(self, in_channels=256):
        super().__init__()
        self.pointConv = nn.Sequential(nn.Conv2d(in_channels, in_channels//4, kernel_size=1, bias=False),
                                        SynchronizedBatchNorm2d(in_channels//4),
                                        nn.ReLU(inplace=True)
                                        )
        pool_scales=(1, 2, 3, 6)
        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
            ))
        self.ppm = nn.ModuleList(self.ppm)

        self.conv_last = nn.Sequential(
            nn.Conv2d(in_channels+len(pool_scales)*(in_channels//4), in_channels,
                      kernel_size=3, padding=1, bias=False),
            SynchronizedBatchNorm2d(in_channels),
            # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            # nn.Conv2d(256, in_channels, kernel_size=1)
        )

        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input):
        input_size = input.size()
        x = self.pointConv(input)

        ppm_out = [input]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.upsample(
                pool_scale(x),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        ppm_out = self.conv_last(ppm_out)

        x = torch.mean(ppm_out, 3, keepdim=True)
        x = torch.mean(x, 2 ,keepdim=True)
        x = self.conv(x)
        x = self.bn(x)
        x = self.sigmoid(x)

        x = torch.mul(ppm_out, x)

        return x

class BiSeNet(torch.nn.Module):
    def __init__(self, num_classes=14, context_path='resnet18', use_softmax=False):
        super().__init__()
        # build spatial path
        self.saptial_path = Spatial_path()

        # build context path
        self.context_path = build_contextpath(name=context_path)
        # self.context_path = AlignedXception(BatchNorm=nn.BatchNorm2d, pretrained=True, output_stride=16)
        # self.context_path = xception(pretrained=False)

        # build attention refinement module
        # self.attention_refinement_module1 = AttentionRefinementModule(1024, 1024)
        # self.attention_refinement_module2 = AttentionRefinementModule(2048, 2048)
        # self.attention_refinement_module1 = AttentionRefinementModule(256, 256)
        # self.attention_refinement_module2 = AttentionRefinementModule(512, 512)

        self.attention_refinement_module1 = PSPModule(256)
        self.attention_refinement_module2 = PSPModule(512)


        # build feature fusion module
        self.feature_fusion_module = FeatureFusionModule(num_classes)

        self.psp = PSPModule(num_classes)

        # build final convolution
        self.conv = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1)

        self.use_softmax = use_softmax

    def forward(self, input, segSize=None):
        # print("x.size=", input.size())
        # output of spatial path
        sx = self.saptial_path(input)
        # print("sx.size=", sx.size())

        # output of context path
        _, cx1, cx2, tail = self.context_path(input) # _ is 1/8, used as deep supervision loss
        # print("cx1.size=", cx1.size())
        # print("cx2.size=", cx2.size())
        # print("tail.size=", tail.size())
        cx1 = self.attention_refinement_module1(cx1)
        cx2 = self.attention_refinement_module2(cx2)
        cx2 = torch.mul(cx2, tail)
        # upsampling
        cx1 = torch.nn.functional.upsample(cx1, scale_factor=2, mode='bilinear', align_corners=False)
        cx2 = torch.nn.functional.upsample(cx2, scale_factor=4, mode='bilinear', align_corners=False)
        cx = torch.cat((cx1, cx2), dim=1)

        # output of feature fusion module
        result = self.feature_fusion_module(sx, cx)

        '''
        here build multi-scale global pooling structure, like PSPNet, plus conv 1*1
        '''
        result = self.psp(result)


        if self.use_softmax:
            # upsampling
            result = torch.nn.functional.upsample(result, scale_factor=8, mode='bilinear', align_corners=False)
            # result = self.conv(result)
            result = nn.functional.softmax(result, dim=1)
            return result

        # result = torch.nn.functional.upsample(result, scale_factor=8, mode='bilinear', align_corners=False)
        result = nn.functional.log_softmax(result, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)
        return result, _

############BiSeNet#########BiSeNet#####BiSeNet#######BiSeNet###########BiSeNet########BiSeNet#############



######## 2222222222222222222222222222222222222222222222222########
'''
class BiseNet(nn.Module):
    """docstring for BiseNet"""
    def __init__(self, input_size=(480, 640), class_num=14, use_softmax=False):
        super().__init__()
        self.use_softmax = use_softmax
        self.spatial_path = SpatialPath()
        self.context_path = ContextPath(input_size)
        self.ffm = FFM(input_size, 1152)
        self.pred = nn.Conv2d(1152, class_num, kernel_size=1, stride=1)

    def forward(self, x, segSize):
        # print("input: ", x.size())
        x1 = self.spatial_path(x)
        x2 = self.context_path(x)
        feature = self.ffm(x1, x2)
        seg = self.pred(feature)
        if self.use_softmax:
            # x = F.upsample(seg, scale_factor=8, mode='bilinear', align_corners=False)
            x = F.upsample(seg, size=(480, 640), mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        x = nn.functional.log_softmax(seg, dim=1)
        return x
        

class SpatialPath(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = self.downsample_block(3, 64)
        self.layer2 = self.downsample_block(64, 128)
        self.layer3 = self.downsample_block(128, 256)

    def downsample_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.layer3(x)


class ARM(nn.Module):

    def __init__(self, size, channels):
        super().__init__()
        self.pool = nn.AvgPool2d(size[0], size[1]) # AdaptiveAvgPool2d(scale)
        # self.pool = nn.AdaptiveAvgPool2d(scale)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, stride=1)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x):
        feature_map = x
        x = self.pool(x)
        x = self.conv(x)
        x = self.norm(x)
        x = torch.sigmoid(x)
        # print(feature_map.size())
        return x.expand_as(feature_map) * feature_map


class FFM(nn.Module):

    def __init__(self, size, channels):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.pool = nn.AvgPool2d(size[0] // 8, size[1] // 8)
        # self.pool = nn.AdaptiveAvgPool2d(8)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1, stride=1)

    def forward(self, x1, x2):
        feature = torch.cat([x1, x2], dim=1)
        feature = self.feature(feature)

        x = self.pool(feature)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = torch.sigmoid(x)
        return feature + x.expand_as(feature) * feature


class ContextPath(nn.Module):

    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.backbone = resnet18(pretrained=True)
        self.x8_arm = ARM((input_size[0] // 8, input_size[1] // 8), 128)
        self.x16_arm = ARM((input_size[0] // 16, input_size[1] // 16), 256)
        self.x32_arm = ARM((input_size[0] // 32, input_size[1] // 32), 512)
        # self.x8_arm = ARM(8, 128)
        # self.x16_arm = ARM(16, 256)
        # self.x32_arm = ARM(32, 512)
        self.global_pool = nn.AvgPool2d(input_size[0] // 32, input_size[1] // 32)
        # self.global_pool = nn.AdaptiveAvgPool2d(32)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)

        feature_x8 = self.backbone.layer2(x)
        feature_x16 = self.backbone.layer3(feature_x8)
        feature_x32 = self.backbone.layer4(feature_x16)
        center = self.global_pool(feature_x32)
        # print("center:", center.size())

        feature_x8 = self.x8_arm(feature_x8)
        feature_x16 = self.x16_arm(feature_x16)
        feature_x32 = self.x32_arm(feature_x32)

        # print("self.input_size[0], self.input_size[1]:", self.input_size[0], self.input_size[1])
        # up_feature_x32 = F.interpolate(center, scale_factor=(self.input_size // 32), mode='bilinear', align_corners=False)
        up_feature_x32 = F.upsample(center, size=(self.input_size[0] // 32, self.input_size[1] // 32), mode='bilinear', align_corners=False)
        # print("feature_x32: ", feature_x32.size())
        # print("up_feature_x32: ", up_feature_x32.size())
        ensemble_feature_x32 = feature_x32 + up_feature_x32

        # up_feature_x16 = F.interpolate(ensemble_feature_x32, scale_factor=2, mode='bilinear', align_corners=False)
        up_feature_x16 = F.upsample(ensemble_feature_x32, scale_factor=2, mode='bilinear', align_corners=False)
        ensemble_feature_x16 = torch.cat((feature_x16, up_feature_x16), dim=1)

        # up_feature_x8 = F.interpolate(ensemble_feature_x16, scale_factor=2, mode='bilinear', align_corners=False)
        up_feature_x8 = F.upsample(ensemble_feature_x16, scale_factor=2, mode='bilinear', align_corners=False)
        ensemble_feature_x8 = torch.cat((feature_x8, up_feature_x8), dim=1)

        return ensemble_feature_x8

'''
######## 2222222222222222222222222222222222222222222222222########


class MacnetV2(nn.Module):
    def __init__(self, n_class=14, in_size=(480, 640), width_mult=1.,
                 out_sec=256, aspp_sec=(12, 24, 36), use_softmax=False):

        super(MacnetV2, self).__init__()

        self.use_softmax = use_softmax
        self.n_class = n_class
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s, d
            # [1, 32, 1, 1, 1],    # 1/2
            # [6, 64, 2, 2, 1],    # 1/4
            # [6, 128, 3, 2, 1],    # 1/8
            # [6, 128, 4, 1, 2],    # 1/8
            # [6, 256, 3, 1, 4],    # 1/8
            # [6, 256, 3, 1, 8],   # 1/8
            # [6, 512, 1, 1, 16],  # 1/8
            [1, 16, 1, 1, 1],    # 1/2
            [6, 24, 2, 2, 1],    # 1/4
            [6, 32, 3, 2, 1],    # 1/8
            [6, 64, 4, 1, 2],    # 1/8
            [6, 96, 3, 1, 4],    # 1/8
            [6, 160, 3, 1, 8],   # 1/8
            [6, 320, 1, 1, 16],  # 1/8
        ]

        # building first layer
        assert in_size[0] % 8 == 0
        assert in_size[1] % 8 == 0

        self.input_size = in_size

        input_channel = int(32 * width_mult)
        self.mod1 = nn.Sequential(OrderedDict([("conv1", conv_bn(inp=3, oup=input_channel, stride=2))]))

        # building inverted residual blocks
        mod_id = 0
        for t, c, n, s, d in self.interverted_residual_setting:
            output_channel = int(c * width_mult)

            # Create blocks for module
            blocks = []
            for block_id in range(n):
                if block_id == 0 and s == 2:
                    blocks.append(("block%d" % (block_id + 1), InvertedResidual(inp=input_channel,
                                                                                oup=output_channel,
                                                                                stride=s,
                                                                                dilate=1,
                                                                                expand_ratio=t)))
                else:
                    blocks.append(("block%d" % (block_id + 1), InvertedResidual(inp=input_channel,
                                                                                oup=output_channel,
                                                                                stride=1,
                                                                                dilate=d,
                                                                                expand_ratio=t)))

                input_channel = output_channel

            self.add_module("mod%d" % (mod_id + 2), nn.Sequential(OrderedDict(blocks)))
            mod_id += 1

        # building last several layers
        org_last_chns = (self.interverted_residual_setting[0][1] +
                         self.interverted_residual_setting[1][1] +
                         self.interverted_residual_setting[2][1] +
                         self.interverted_residual_setting[3][1] +
                         self.interverted_residual_setting[4][1] +
                         self.interverted_residual_setting[5][1] +
                         self.interverted_residual_setting[6][1])

        self.last_channel = int(org_last_chns * width_mult) if width_mult > 1.0 else org_last_chns
        self.out_se = nn.Sequential(SCSEBlock(self.last_channel, reduction=16))

        # if self.n_class != 0:
        #     self.aspp = nn.Sequential(ASPPBlock(self.last_channel, out_sec))

        #     in_stag2_up_chs = self.interverted_residual_setting[1][1] + self.interverted_residual_setting[0][1]
        #     self.score_se = nn.Sequential(SCSEBlock((out_sec + in_stag2_up_chs), reduction=16))
        #     self.score = nn.Sequential(OrderedDict([("norm_1", nn.BatchNorm2d(out_sec + in_stag2_up_chs)),
        #                                             ("conv_1", nn.Conv2d(out_sec + in_stag2_up_chs,
        #                                                                  out_sec + in_stag2_up_chs,
        #                                                                  kernel_size=3, stride=1, padding=2,
        #                                                                  dilation=2, bias=False)),
        #                                             ("norm_2", nn.BatchNorm2d(out_sec + in_stag2_up_chs)),
        #                                             ("conv_2", nn.Conv2d(out_sec + in_stag2_up_chs, self.n_class,
        #                                                                  kernel_size=1, stride=1, padding=0,
        #                                                                  bias=True))])
        #                                             )
        #     # self.upsampling = nn.Upsample(size=(480, 640), mode='bilinear')

        self._initialize_weights()


        #####  ppm module   #######
        num_class=14
        fc_dim=self.last_channel
        pool_scales=(1, 2, 3, 6)
        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(512),
                # nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)
        fc_dim_sup=self.interverted_residual_setting[6][1]
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim_sup // 2, fc_dim_sup // 4, 1)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            SynchronizedBatchNorm2d(512),
            # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )

        self.conv_last_deepsup = nn.Conv2d(fc_dim_sup // 4, num_class, 1, 1, 0)
        self.dropout_deepsup = nn.Dropout2d(0.1)

        ########  finish ppm module  #######

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x, segSize=None):
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 1. Encoder: feature extraction
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        stg1 = self.mod1(x)     # (N, 32,   224, 448)  1/2
        stg1 = self.mod2(stg1)  # (N, 32,   224, 448)  1/2 -> 1/4 -> 1/8
        stg2 = self.mod3(stg1)  # (N, 64,   112, 224)  1/4 -> 1/8
        stg3 = self.mod4(stg2)  # (N, 128,   56,  112)  1/8
        stg4 = self.mod5(stg3)  # (N, 128,   56,  112)  1/8 dilation=2
        stg5 = self.mod6(stg4)  # (N, 256,   56,  112)  1/8 dilation=4
        stg6 = self.mod7(stg5)  # (N, 256,  56,  112)  1/8 dilation=8
        stg7 = self.mod8(stg6)  # (N, 512,  56,  112)  1/8 dilation=16

        stg1_1 = F.max_pool2d(input=stg1, kernel_size=3, stride=2, ceil_mode=True)    # 1/4
        stg1_2 = F.max_pool2d(input=stg1_1, kernel_size=3, stride=2, ceil_mode=True)  # 1/8
        stg2_1 = F.max_pool2d(input=stg2, kernel_size=3, stride=2, ceil_mode=True)    # 1/8

        # (N, 672, 56,  112)  1/8  (16+24+32+64+96+160+320)
        stg8 = self.out_se(torch.cat([stg3, stg4, stg5, stg6, stg7, stg1_2, stg2_1], dim=1))


        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 1. Decoder: ppm module
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        conv5 = stg8

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.upsample(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.upsample(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        # deep sup
        conv4 = stg6
        _ = self.cbr_deepsup(conv4)
        _ = self.dropout_deepsup(_)
        _ = self.conv_last_deepsup(_)

        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)

        return (x, _)

        '''
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 2. Decoder: multi-scale feature fusion
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        if self.n_class != 0:
            # (N, 672, H/8, W/8) -> (N, 256, H/4, W/4)
            de_stg1 = self.aspp(stg8)

            # (N, 256+24+16=296, H/4, W/4)
            de_stg1 = self.score_se(torch.cat([de_stg1, stg2, stg1_1], dim=1))

            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
            # 3. Classifier: pixel-wise classification-segmentation
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
            net_out = self.score(de_stg1)

            net_out = nn.functional.upsample(net_out, size=(480, 640), mode='bilinear', align_corners=False)
            if self.use_softmax: # is True during inference
                net_out = nn.functional.log_softmax(net_out, dim=1)
            else:
                net_out = nn.functional.softmax(net_out, dim=1)

            return net_out
        else:
            return stg8
    
        '''


class Macnet(nn.Module):
    def __init__(self, orig_mobilenet, num_class=14, fc_dim=256, use_softmax=False):
        super(Macnet, self).__init__()
        self.use_softmax = use_softmax
        #encoder
        self.conv1 = orig_mobilenet.conv1  # 1/2 32
        self.bn1 = orig_mobilenet.bn1
        self.relu = orig_mobilenet.relu

        self.layer1 = orig_mobilenet.layer1
        self.layer2 = orig_mobilenet.layer2 # 1/4 32
        self.layer3 = orig_mobilenet.layer3 # 1/8 64
        self.layer4 = orig_mobilenet.layer4
        self.layer5 = orig_mobilenet.layer5

        self.pwconv = nn.Conv2d(256, 64, 1, 1, 0)
        self.bn2 = SynchronizedBatchNorm2d(64)

        #decoder
        self.up_1 = nn.Sequential(nn.ConvTranspose2d((64+64), 64, 2, stride=2, padding=0, output_padding=0, bias=False), BR(64))
        self.up_2 = nn.Sequential(nn.ConvTranspose2d((64+32), 32, 2, stride=2, padding=0, output_padding=0, bias=False), BR(32))
        self.up_3 = nn.Sequential(nn.ConvTranspose2d((32+32), 14, 2, stride=2, padding=0, output_padding=0, bias=False), BR(14))


    def forward(self, x, segSize=None):
        x = self.conv1(x)
        x = self.bn1(x)
        out_1 = self.relu(x)

        x = self.layer1(out_1)
        out_2 = self.layer2(x)
        out_3 = self.layer3(out_2)
        x = self.layer4(out_3)
        x = self.layer5(x)

        x = self.pwconv(x)
        x = self.bn2(x)

        x = torch.cat([x, out_3], dim=1)
        x = self.up_1(x)
        x = torch.cat([x, out_2], dim=1)
        x = self.up_2(x)
        x = torch.cat([x, out_1], dim=1)
        x = self.up_3(x)

        if self.use_softmax: # is True during inference
            x = nn.functional.upsample(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)

        return x


class Resnet_esp(nn.Module):
    def __init__(self, orig_resnet, p=5, q=3):
        super(Resnet_esp, self).__init__()

        self.pwconv = nn.Conv2d(128+3, 128, kernel_size=1, bias=False)

        self.sample1 = InputProjectionA(1)
        self.sample2 = InputProjectionA(1)

        self.b1 = BR(64 + 3)
        self.level2_0 = DownSamplerB(64 +3, 64)

        self.level2 = nn.ModuleList()
        for i in range(0, p):
            self.level2.append(DilatedParllelResidualBlockB(64 , 64))
        self.b2 = BR(128 + 3)

        self.level3_0 = DilatedParllelResidualBlockB(128 + 3, 128)
        self.level3 = nn.ModuleList()
        for i in range(0, q):
            self.level3.append(DilatedParllelResidualBlockB(128 , 128))
        self.b3 = BR(256)

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = conv3x3(3, 64)
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = conv3x3(256, 128)
        self.bn2 = SynchronizedBatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool = orig_resnet.maxpool

        from functools import partial
        orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
        orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        inp1 = self.sample1(x)  # to be concated 
        inp2 = self.sample2(x)  # to be concated 

        output0 = self.relu1(self.bn1(self.conv1(x))) #   3->64
        output0_cat = self.b1(torch.cat([output0, x], 1))   # 64->67
        output1_0 = self.level2_0(output0_cat) # down-sampled 67->64

        # 64->64 same resolution
        for i, layer in enumerate(self.level2):
            if i==0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)
        output1_cat = self.b2(torch.cat([output1,  output1_0, inp1], 1)) # 64->128+3

        output2_0 = self.pwconv(output1_cat) # 128+3 -> 128 ->128
        for i, layer in enumerate(self.level3):
            if i==0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        output2_cat = self.b3(torch.cat([output2_0, output2], 1))  # 128->256

        # 256->128
        x = self.relu2(self.bn2(self.conv2(output2_cat)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x); # 2x
        x = self.layer2(x); conv_out.append(x); # 2x
        x = self.layer3(x); conv_out.append(x); # 2x
        x = self.layer4(x); conv_out.append(x); # 2x

        if return_feature_maps:
            return conv_out
        return [x]


######## mobilenet for encoder, juest for testing  ########

class MobilenetV2(nn.Module):
    def __init__(self, orig_mobilenet):
        super(MobilenetV2, self).__init__()
        self.conv1 = orig_mobilenet.conv1
        self.bn1 = orig_mobilenet.bn1
        self.relu = orig_mobilenet.relu

        self.layer1 = orig_mobilenet.layer1
        self.layer2 = orig_mobilenet.layer2
        self.layer3 = orig_mobilenet.layer3
        self.layer4 = orig_mobilenet.layer4
        self.layer5 = orig_mobilenet.layer5
        self.layer6 = orig_mobilenet.layer6
        self.layer7 = orig_mobilenet.layer7

        self.conv8 = orig_mobilenet.conv8

    def forward(self, x, return_feature_maps=False):
        conv_out = []
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x);conv_out.append(x);
        x = self.layer6(x);conv_out.append(x);
        x = self.layer7(x);conv_out.append(x);

        x = self.conv8(x);conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]


class Mobilenet(nn.Module):
    def __init__(self, orig_mobilenet):
        super(Mobilenet, self).__init__()

        # take pretrained mobilenet, except AvgPool and FC
        self.conv1 = orig_mobilenet.conv_bn_inp_32_2
        self.conv_dw_1 = orig_mobilenet.conv_dw_32_64_1
        self.conv_dw_2 = orig_mobilenet.conv_dw_64_128_2
        self.conv_dw_3 = orig_mobilenet.conv_dw_128_128_1
        self.conv_dw_4 = orig_mobilenet.conv_dw_128_256_2
        self.conv_dw_5 = orig_mobilenet.conv_dw_256_256_1
        self.conv_dw_6 = orig_mobilenet.conv_dw_256_512_2
        self.conv_dw_7 = orig_mobilenet.conv_dw_512_512_1
        self.conv_dw_8 = orig_mobilenet.conv_dw_512_1024_2
        self.conv_dw_9 = orig_mobilenet.conv_dw_1024_oup_1


    def forward(self, x, return_feature_maps=False):
        conv_out = []
        x = self.conv1(x)
        x = self.conv_dw_1(x)
        x = self.conv_dw_2(x)
        x = self.conv_dw_3(x)
        x = self.conv_dw_4(x)
        x = self.conv_dw_5(x); conv_out.append(x);
        x = self.conv_dw_6(x)
        x = self.conv_dw_7(x); conv_out.append(x);
        x = self.conv_dw_8(x); conv_out.append(x);
        x = self.conv_dw_9(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]


class Resnet(nn.Module):
    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]


class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResnetDilated, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        # import pdb
        # pdb.set_trace()

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]


# last conv, bilinear upsample
class C1BilinearDeepSup(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1BilinearDeepSup, self).__init__()
        self.use_softmax = use_softmax

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        x = self.cbr(conv5)
        x = self.conv_last(x)

        if self.use_softmax:  # is True during inference
            x = nn.functional.upsample(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        # deep sup
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.conv_last_deepsup(_)

        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)

        return (x, _)


# last conv, bilinear upsample
class C1Bilinear(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1Bilinear, self).__init__()
        self.use_softmax = use_softmax

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        x = self.cbr(conv5)
        x = self.conv_last(x)

        if self.use_softmax: # is True during inference
            x = nn.functional.upsample(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)

        return x


# pyramid pooling, bilinear upsample
class PPMBilinear(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPMBilinear, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            SynchronizedBatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.upsample(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.upsample(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)
        return x


# pyramid pooling, bilinear upsample
class PPMBilinearDeepsup(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPMBilinearDeepsup, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            SynchronizedBatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )

        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.dropout_deepsup = nn.Dropout2d(0.1)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.upsample(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.upsample(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        # deep sup
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.dropout_deepsup(_)
        _ = self.conv_last_deepsup(_)

        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)

        return (x, _)

# upernet
class UPerNet(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6),
                 fpn_inplanes=(256,512,1024,2048), fpn_dim=256):
        super(UPerNet, self).__init__()
        self.use_softmax = use_softmax

        # PPM Module
        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(nn.Sequential(
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(fc_dim + len(pool_scales)*512, fpn_dim, 1)

        # FPN Module
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]: # skip the top layer
            self.fpn_in.append(nn.Sequential(
                nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ))
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1): # skip the top layer
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(nn.functional.upsample(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False)))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x) # lateral branch

            f = nn.functional.upsample(
                f, size=conv_x.size()[2:], mode='bilinear', align_corners=False) # top-down branch
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse() # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.upsample(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.upsample(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        x = nn.functional.log_softmax(x, dim=1)

        return x

class CBR(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''
    def __init__(self, nIn, nOut, kSize, stride=1):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super(CBR, self).__init__()
        padding = int((kSize - 1)/2)
        #self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        #self.conv1 = nn.Conv2d(nOut, nOut, (1, kSize), stride=1, padding=(0, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        #output = self.conv1(output)
        output = self.bn(output)
        output = self.act(output)
        return output


class BR(nn.Module):
    '''
        This class groups the batch normalization and PReLU activation
    '''
    def __init__(self, nOut):
        '''
        :param nOut: output feature maps
        '''
        super(BR, self).__init__()
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: normalized and thresholded feature map
        '''
        output = self.bn(input)
        output = self.act(output)
        return output

class CB(nn.Module):
    '''
       This class groups the convolution and batch normalization
    '''
    def __init__(self, nIn, nOut, kSize, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optinal stide for down-sampling
        '''
        super(CB, self).__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)

    def forward(self, input):
        '''

        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        output = self.bn(output)
        return output

class C(nn.Module):
    '''
    This class is for a convolutional layer.
    '''
    def __init__(self, nIn, nOut, kSize, stride=1):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        '''
        super(C, self).__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output

class CDilated(nn.Module):
    '''
    This class defines the dilated convolution.
    '''
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super(CDilated, self).__init__()
        padding = int((kSize - 1)/2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False, dilation=d)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output

class DownSamplerB(nn.Module):
    def __init__(self, nIn, nOut):
        super(DownSamplerB, self).__init__()
        n = int(nOut/5)
        n1 = nOut - 4*n
        self.c1 = C(nIn, n, 3, 2)
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 4)
        self.d8 = CDilated(n, n, 3, 1, 8)
        self.d16 = CDilated(n, n, 3, 1, 16)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-3)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        combine = torch.cat([d1, add1, add2, add3, add4],1)
        #combine_in_out = input + combine
        output = self.bn(combine)
        output = self.act(output)
        return output

class DilatedParllelResidualBlockB(nn.Module):
    '''
    This class defines the ESP block, which is based on the following principle
        Reduce ---> Split ---> Transform --> Merge
    '''
    def __init__(self, nIn, nOut, add=True):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param add: if true, add a residual connection through identity operation. You can use projection too as
                in ResNet paper, but we avoid to use it if the dimensions are not the same because we do not want to
                increase the module complexity
        '''
        super(DilatedParllelResidualBlockB, self).__init__()
        n = int(nOut/5)
        n1 = nOut - 4*n
        self.c1 = C(nIn, n, 1, 1)
        self.d1 = CDilated(n, n1, 3, 1, 1) # dilation rate of 2^0
        self.d2 = CDilated(n, n, 3, 1, 2) # dilation rate of 2^1
        self.d4 = CDilated(n, n, 3, 1, 4) # dilation rate of 2^2
        self.d8 = CDilated(n, n, 3, 1, 8) # dilation rate of 2^3
        self.d16 = CDilated(n, n, 3, 1, 16) # dilation rate of 2^4
        self.bn = BR(nOut)
        self.add = add

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        # reduce
        output1 = self.c1(input)
        # split and transform
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        # heirarchical fusion for de-gridding
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        #merge
        combine = torch.cat([d1, add1, add2, add3, add4], 1)

        # if residual version
        if self.add:
            combine = input + combine
        output = self.bn(combine)
        return output

class InputProjectionA(nn.Module):
    '''
    This class projects the input image to the same spatial dimensions as the feature map.
    For example, if the input image is 512 x512 x3 and spatial dimensions of feature map size are 56x56xF, then
    this class will generate an output of 56x56x3
    '''
    def __init__(self, samplingTimes):
        '''
        :param samplingTimes: The rate at which you want to down-sample the image
        '''
        super(InputProjectionA, self).__init__()
        self.pool = nn.ModuleList()
        for i in range(0, samplingTimes):
            #pyramid-based approach for down-sampling
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        '''
        :param input: Input RGB Image
        :return: down-sampled image (pyramid-based approach)
        '''
        for pool in self.pool:
            input = pool(input)
        return input


class ESPNet_Encoder(nn.Module):
    '''
    This class defines the ESPNet-C network in the paper
    '''
    def __init__(self, classes=14, p=5, q=3):
        '''
        :param classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param p: depth multiplier
        :param q: depth multiplier
        '''
        super(ESPNet_Encoder, self).__init__()
        self.level1 = CBR(3, 16, 3, 2)
        self.sample1 = InputProjectionA(1)
        self.sample2 = InputProjectionA(2)

        self.b1 = BR(16 + 3)
        self.level2_0 = DownSamplerB(16 +3, 64)

        self.level2 = nn.ModuleList()
        for i in range(0, p):
            self.level2.append(DilatedParllelResidualBlockB(64 , 64))
        self.b2 = BR(128 + 3)

        self.level3_0 = DownSamplerB(128 + 3, 128)
        self.level3 = nn.ModuleList()
        for i in range(0, q):
            self.level3.append(DilatedParllelResidualBlockB(128 , 128))
        self.b3 = BR(256)

        self.classifier = C(256, classes, 1, 1)

    def forward(self, input, return_feature_maps=False):
        '''
        :param input: Receives the input RGB image
        :return: the transformed feature map with spatial dimensions 1/8th of the input image
        '''

        output0 = self.level1(input)   # 2x
        inp1 = self.sample1(input)
        inp2 = self.sample2(input)

        output0_cat = self.b1(torch.cat([output0, inp1], 1))
        output1_0 = self.level2_0(output0_cat) # down-sampled
        
        for i, layer in enumerate(self.level2):
            if i==0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.b2(torch.cat([output1,  output1_0, inp2], 1))

        output2_0 = self.level3_0(output1_cat) # down-sampled
        for i, layer in enumerate(self.level3):
            if i==0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        output2_cat = self.b3(torch.cat([output2_0, output2], 1))

        classifier = self.classifier(output2_cat)

        if return_feature_maps:
            return conv_out
        return classifier
############   END  FOR  TESTING    ####################    