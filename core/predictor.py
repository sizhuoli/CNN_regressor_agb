#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 17:27:20 2022

@author: sizhuo
"""

from torch.backends import cudnn
import random
import matplotlib.pyplot as plt  # plotting tools
import numpy as np
import glob
import json
import time
import pandas as pd
import os
from tqdm import tqdm
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import rasterio
from core.evaluation import *
from torchinfo import summary
from itertools import product
from rasterio import windows
from torchvision import transforms as T
from rasterio.enums import Resampling
from scipy.ndimage import zoom
import ipdb
import torchvision.models as models




class Effnet_2inputs(nn.Module):
    # efficientnet with 2 inputs, one branch with pretrained weights
    def __init__(self, net):
        super(Effnet_2inputs,self).__init__()

        self.im_features = nn.Sequential(*list(net.features.children()), net.avgpool)
        num_feat = self.im_features[-2][0].out_channels
        self.dropout = nn.Dropout(p=0.2)
        # for input2
        self.dense1 = nn.Linear(in_features = 1, out_features = 32, bias=True)
        self.dense2 = nn.Linear(in_features = 32, out_features = 64, bias=True)

        self.im_dense1 = nn.Linear(in_features=1280, out_features=640, bias=True)
        self.final_dense1 = nn.Linear(in_features=640+64, out_features=320, bias=True)
        self.final_dense2 = nn.Linear(in_features=320, out_features=64, bias=True)
        self.final_dense3 = nn.Linear(in_features=64, out_features=1, bias=True)


    def forward(self, img, input2):

        f1 = self.im_features(img) # feature extracted from image branch # 32, 1280
        f1 = f1.view(f1.size(0), -1) #(32, 1280, 1, 1) to (32, 1280)
        f2 = self.im_dense1(f1) # 32, 640

        # additional input2
        # import ipdb
        # ipdb.set_trace()
        a1 = self.dense1(input2.view(input2.size(0), -1))
        a1 = self.dropout(a1)
        a2 = self.dense2(a1)
        a2 = self.dropout(a2)

        # import ipdb
        # ipdb.set_trace()
        # combine two inputs
        combined = torch.cat((f2.view(f2.size(0), -1),
                          a2.view(a2.size(0), -1)), dim=1)

        c1 = self.final_dense1(combined)
        c1 = self.dropout(c1)
        c2 = self.final_dense2(c1)
        c2 = self.dropout(c2)
        c3 = self.final_dense3(c2)


        return c3


class Effnet_3outputs_AGB_B_C(nn.Module):
    # efficientnet with 3 outputs
    # AGB: biomass
    # B broadleaved fraction
    # C coniferous fraction
    def __init__(self, net):
        super(Effnet_3outpuefficientb0ts_AGB_B_C,self).__init__()

        self.im_features = nn.Sequential(*list(net.features.children()), net.avgpool)
        num_feat = self.im_features[-2][0].out_channels
        self.dropout = nn.Dropout(p=0.2)

        self.dense0 = nn.Linear(in_features=1280, out_features=640, bias=True)
        # branches having equal dim
        self.dense1 = nn.Linear(in_features=640, out_features=320, bias=True)
        self.dense2 = nn.Linear(in_features=320, out_features=64, bias=True)
        self.dense3 = nn.Linear(in_features=64, out_features=1, bias=True)



    def forward(self, img):

        f1 = self.im_features(img) # feature extracted from image branch # 32, 1280
        f1 = f1.view(f1.size(0), -1) #(32, 1280, 1, 1) to (32, 1280)
        f2 = self.dense0(f1) # 32, 640
        f2 = self.dropout(f2)
        im1 = self.dense1(f2) # 32, 320
        im1 = self.dropout(im1)
        im2 = self.dense2(im1) #32, 64
        im2 = self.dropout(im2)
        im3 = self.dense3(im2) #32, 1

        bfr1 = self.dense2(im1) #32, 64
        bfr1 = self.dropout(bfr1)
        bfr2 = self.dense3(bfr1) #32, 1

        cfr1 = self.dense2(im1) #32, 64
        cfr1 = self.dropout(cfr1)
        cfr2 = self.dense3(cfr1) #32, 1

        return im3, bfr2, cfr2



class Effnet_2outputs_AGB_activMap(nn.Module):
    # inherent from regnet map model_
    # efficientnet with 2 outputs
    # AGB: biomass
    # activation map
    def __init__(self, net, arch = 'reg'):
        super(Effnet_2outputs_AGB_activMap,self).__init__()
        # ipdb.set_trace()
        # self.im_features = nn.Sequential(*list(net.stem.children()), *list(net.trunk_output.children()))
        if arch == 'eff':
            # regnet arch
            self.im_features = nn.Sequential(*list(net.features.children())[:-10])

            nf = net.features[-11][-1].block[-1][0].out_channels
            # b2 arch has 120 kernels
            # ipdb.set_trace()
            self.im_features.append(nn.Conv2d(nf, 64, kernel_size=(1, 1), stride=(1, 1), bias=False))
            self.im_features.append(nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
            self.im_features.append(nn.Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False))
            self.im_features.append(nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
            self.im_features.append(nn.Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1), bias=True))
            # # enforce non-negative preds
            # self.im_features.append(nn.ReLU(inplace=True))

        else:
            # effb arch
            self.im_features = nn.Sequential(*list(net.stem.children()), *list(net.trunk_output.children())[:-8]) # down to 14*14
            self.im_features.append(nn.Conv2d(320, 240, kernel_size=(1, 1), stride=(1, 1), bias=False))

            self.im_features.append(nn.BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
            self.im_features.append(nn.Conv2d(240, 64, kernel_size=(1, 1), stride=(1, 1), bias=False))
            self.im_features.append(nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
            self.im_features.append(nn.Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False))
            self.im_features.append(nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
            self.im_features.append(nn.Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1), bias=True))


    def forward(self, img):

        f1 = self.im_features(img) # feature extracted from image branch # 32, 1280

        # enforce non-negative preds
        # ipdb.set_trace()
        # m = nn.ReLU(inplace = False)
        # f2 = m(f1)
        # .unsqueeze(-1)
        return f1

class Effnet_3outputs_AGB_F_B(nn.Module):
    # efficientnet with 3 outputs
    # F: whether forest > 0 (biomass > 0) or not, whether contains trees
    # AGB: biomass
    # B broadleaved fraction

    def __init__(self, net):
        super(Effnet_3outputs_AGB_F_B,self).__init__()

        self.im_features = nn.Sequential(*list(net.features.children()), net.avgpool)
        num_feat = self.im_features[-2][0].out_channels
        self.dropout = nn.Dropout(p=0.2)

        self.dense0 = nn.Linear(in_features=1280, out_features=640, bias=True)
        # branches having equal dim
        self.dense1 = nn.Linear(in_features=640, out_features=320, bias=True)
        self.dense2 = nn.Linear(in_features=320, out_features=64, bias=True)
        self.dense3 = nn.Linear(in_features=64, out_features=1, bias=True)

    def forward(self, img):

        f1 = self.im_features(img) # feature extracted from image branch # 32, 1280
        f1 = f1.view(f1.size(0), -1) #(32, 1280, 1, 1) to (32, 1280)
        f2 = self.dense0(f1) # 32, 640
        f2 = self.dropout(f2)
        im1 = self.dense1(f2) # 32, 320
        im1 = self.dropout(im1)
        im2 = self.dense2(im1) #32, 64
        im2 = self.dropout(im2)
        im3 = self.dense3(im2) #32, 1

        f1 = self.dense2(im1) #32, 64
        f1 = self.dropout(f1)
        f2 = self.dense3(f1) #32, 1

        b1 = self.dense2(im1) #32, 64
        b1 = self.dropout(b1)
        b2 = self.dense3(b1) #32, 1

        return im3, f2, b2 # AGB, F, B

class Effnet_AGB_activMap_attention(nn.Module):
    def __init__(self, net, in_c):
        super(Effnet_AGB_activMap_attention,self).__init__()
        # ipdb.set_trace()
        self.im_features = nn.Sequential(*list(net.stem.children()), *list(net.trunk_output.children()))
        self.atten = nn.Sequential(SpatialAttention2d(in_c=in_c, act_fn='relu'))
        self.wei_atten = nn.Sequential(Weighted2d())

    def forward(self, img):

        f1 = self.im_features(img) # feature extracted
        a1 = self.atten(f1)
        # ipdb.set_trace()
        res = self.wei_atten([f1, a1]) # weighted activation map
        # ipdb.set_trace()
        return res

def copyParams(params_src, params_dest):
    for name, param in params_src.items():
        if name in params_dest:
            print('loading weights and freeze from: ', name)
            params_dest[name].data.copy_(param.data)
            params_dest[name].requires_grad = False

class WeightedSum2d(nn.Module):
    def __init__(self):
        super(WeightedSum2d, self).__init__()
    def forward(self, x):
        x, weights = x
        assert x.size(2) == weights.size(2) and x.size(3) == weights.size(3),\
                'err: h, w of tensors x({}) and weights({}) must be the same.'\
                .format(x.size, weights.size)
        y = x * weights                                       # element-wise multiplication
        y = y.view(-1, x.size(1), x.size(2) * x.size(3))      # b x c x hw
        return torch.sum(y, dim=2).view(-1, x.size(1), 1, 1)  # b x c x 1 x 1
    def __repr__(self):
        return self.__class__.__name__

class Weighted2d(nn.Module):
    # weight while keeping 2d resolution, need to sum for final agb scalar
    def __init__(self):
        super(Weighted2d, self).__init__()
    def forward(self, x):
        # ipdb.set_trace()
        x, weights = x
        assert x.size(2) == weights.size(2) and x.size(3) == weights.size(3),\
                'err: h, w of tensors x({}) and weights({}) must be the same.'\
                .format(x.size, weights.size)
        y = x * weights                                       # element-wise multiplication
        return y
    def __repr__(self):
        return self.__class__.__name__


class SpatialAttention2d(nn.Module):
    '''
    SpatialAttention2d
    2-layer 1x1 conv network with softplus activation.
    <!!!> attention score normalization will be added for experiment.
    '''
    def __init__(self, in_c, act_fn='relu'):
        super(SpatialAttention2d, self).__init__()
        self.conv1 = nn.Conv2d(in_c, 256, 1, 1)                 # 1x1 conv
        if act_fn.lower() in ['relu']:
            self.act1 = nn.ReLU()
        elif act_fn.lower() in ['leakyrelu', 'leaky', 'leaky_relu']:
            self.act1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(256, 1, 1, 1)                    # 1x1 conv
        self.softplus = nn.Softplus(beta=1, threshold=200)       # use default setting.

    def forward(self, x):
        '''
        x : spatial feature map. (b x c x w x h)
        s : softplus attention score
        '''
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.softplus(x)
        return x

    def __repr__(self):
        return self.__class__.__name__

def __freeze_weights__(module_dict, freeze=[]):
    for _, v in enumerate(freeze):
        module = module_dict[v]
        for param in module.parameters():
            param.requires_grad = False

def __print_freeze_status__(model):
    '''print freeze stagus. only for debugging purpose.
    '''
    for i, module in enumerate(model.named_children()):
        for param in module[1].parameters():
            print('{}:{}'.format(module[0], str(param.requires_grad)))


def alpha_loss(loss, alpha = 200):
    alpha_set = loss>alpha
    return loss*(alpha_set+1) # double loss for larger values




class Predictor:
    # inference only
    def __init__(self, config):
        self.config = config
        self.all_files = load_files(config)
        self.model_type = config.model_type
        self.pretrained = config.pretrained
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.weightDecay = config.weightDecay
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # print(self.config)
        # load model and compile
        self.build_model()
        self.load_weights()

        if not os.path.exists(self.config.output_dir):
            os.makedirs(self.config.output_dir)

        # loop through image, crop patches, apply same data transforms

        # crop into 200 p patches, 40 m

    def build_model(self):
        """Build generator and discriminator."""
        if 'torchEfficientnetb0' in self.model_type:
            if self.config.task == 'classification':
                # use default arch
                self.nnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
                # ipdb.set_trace()
                num_fs = self.nnet.classifier[1].in_features
                self.nnet.classifier = torch.nn.Sequential(nn.Dropout(p=0.2, inplace=True),
                                                           nn.Linear(in_features=num_fs, out_features=640, bias=True),
                                                           nn.Dropout(p=0.2, inplace=True),
                                                           nn.Linear(in_features=640, out_features=320, bias=True),
                                                           nn.Dropout(p=0.2, inplace=True),
                                                           nn.Linear(in_features=320, out_features=64, bias=True),
                                                           nn.Dropout(p=0.2, inplace=True),
                                                           nn.Linear(in_features=64, out_features=1, bias=True))
            elif self.config.task == 'regression':
                if not self.pretrained:
                    self.nnet = models.efficientnet_b0(num_classes=self.output_ch)
                    self.nnet.features[0][0] = nn.Conv2d(self.img_ch, 32, kernel_size=(3, 3), stride=(1, 1),
                                                         padding=(1, 1), bias=False)
                    # # self.nnet.features[2][0].block[1][0] = nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=96, bias=False)
                    # self.nnet.features[-1][0] = nn.Conv2d(320, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
                    # self.nnet.features[-1][1] = nn.BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    num_fs = self.nnet.classifier[1].in_features
                    self.nnet.classifier = torch.nn.Sequential(nn.Dropout(p=0.2),
                                                               nn.Linear(in_features=num_fs, out_features=640,
                                                                         bias=True),
                                                               nn.Dropout(p=0.2),
                                                               nn.Linear(in_features=640, out_features=320, bias=True),
                                                               nn.Dropout(p=0.2),
                                                               nn.Linear(in_features=320, out_features=1, bias=True),
                                                               )


                # try generating a map instead
                else:  # load pretrained, only reini the final layer
                    if 'dense' in self.model_type:
                        # load pretrained, only reini the final layer
                        # net0 = models.efficientnet_b0(pretrained=True)
                        net0 = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
                        num_fs = net0.classifier[1].in_features

                        if self.add_input2:
                            # add input 2
                            self.nnet = Effnet_2inputs(net0)

                        elif self.add_outputs:
                            self.nnet = Effnet_3outputs(net0)
                        else:
                            # shrink regression output
                            self.nnet = net0

                            self.nnet.classifier = torch.nn.Sequential(nn.Dropout(p=0.2),
                                                                       nn.Linear(in_features=num_fs,
                                                                                 out_features=640, bias=True),
                                                                       nn.Dropout(p=0.2),
                                                                       nn.Linear(in_features=640, out_features=320,
                                                                                 bias=True),
                                                                       nn.Dropout(p=0.2),
                                                                       nn.Linear(in_features=320, out_features=1,
                                                                                 bias=True),
                                                                       )




                    elif 'map' in self.model_type:
                        if 'nopadding' not in self.model_type:
                            net0 = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

                            # shrink regression output
                            self.nnet = net0
                            self.nnet.features[-1][0] = nn.Conv2d(320, 640, kernel_size=(1, 1), stride=(1, 1),
                                                                  bias=False)
                            self.nnet.features[-1][1] = nn.BatchNorm2d(640, eps=1e-05, momentum=0.1, affine=True,
                                                                       track_running_stats=True)
                            self.nnet.features.append(
                                nn.Conv2d(640, 240, kernel_size=(1, 1), stride=(1, 1), bias=False))
                            self.nnet.features.append(
                                nn.BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                            self.nnet.features.append(nn.Conv2d(240, 64, kernel_size=(1, 1), stride=(1, 1), bias=False))
                            self.nnet.features.append(
                                nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                            self.nnet.features.append(nn.Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False))
                            self.nnet.features.append(
                                nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                            self.nnet.features.append(nn.Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1), bias=True))
                            # self.nnet.features.append(nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                            # self.nnet.features[-1][1] = nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

                            # remove avg poly
                            self.nnet.avgpool = torch.nn.Identity()

                            self.nnet.classifier = torch.nn.Sequential()
                            # ipdb.set_trace()
                            if self.config.add_activ_14reso:  # for visualize, do not loss descend
                                # ipdb.set_trace()
                                self.nnet = Effnet_2outputs_AGB_activMap(self.nnet, arch='eff')
                            # # remove early paddings, cant do it more later layers due to dim mismatch
                            # for module in self.nnet.features[:2].modules():
                            #     if isinstance(module, nn.Conv2d):
                            #         module.padding = (0, 0)

                        else:  # with padding

                            net0 = _efficientnet(inverted_residual_setting, 0.2, last_channel=None,
                                                 weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1, progress=True)

                            # shrink regression output
                            self.nnet = net0
                            # self.nnet.features[-1][0] = nn.Conv2d(320, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
                            # self.nnet.features[-1][1] = nn.BatchNorm2d(640, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                            # self.nnet.features.append(nn.Conv2d(640, 320, kernel_size=(1, 1), stride=(1, 1), bias=False))
                            # self.nnet.features.append(nn.BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                            # self.nnet.features.append(nn.Conv2d(320, 64, kernel_size=(1, 1), stride=(1, 1), bias=False))
                            # self.nnet.features.append(nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                            # self.nnet.features.append(nn.Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1), bias=False))

                            self.nnet.features[-4] = torch.nn.Sequential(
                                nn.Conv2d(80, 64, kernel_size=(1, 1), stride=(1, 1), bias=False),
                                nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                            self.nnet.features[-3] = torch.nn.Sequential(
                                nn.Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False),
                                nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                            self.nnet.features[-2] = torch.nn.Sequential(
                                nn.Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1), bias=False))
                            self.nnet.features[-1] = torch.nn.Sequential()
                            # self.nnet.features.append(nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                            # self.nnet.features[-1][1] = nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                            self.nnet.classifier = torch.nn.Sequential()

                            for module in self.nnet.features.modules():
                                if isinstance(module, nn.Conv2d):
                                    module.padding = (0, 0)

        elif 'torchEfficientnetb1' in self.model_type:

            if self.config.task == 'classification':
                self.nnet = models.efficientnet_b1(pretrained=self.pretrained)
                num_fs = self.nnet.classifier[1].in_features
                self.nnet.classifier = torch.nn.Sequential(nn.Dropout(p=0.2, inplace=True),
                                                           nn.Linear(in_features=num_fs, out_features=640, bias=True),
                                                           nn.Dropout(p=0.2, inplace=True),
                                                           nn.Linear(in_features=640, out_features=320, bias=True),
                                                           nn.Dropout(p=0.2, inplace=True),
                                                           nn.Linear(in_features=320, out_features=64, bias=True),
                                                           nn.Dropout(p=0.2, inplace=True),
                                                           nn.Linear(in_features=64, out_features=1, bias=True))

            elif self.config.task == 'regression':
                print('model architecture loaded for ', self.model_type)
                assert self.pretrained == 1
                self.nnet = models.efficientnet_b1(pretrained=self.pretrained)
                if 'dense' in self.model_type:
                    num_fs = self.nnet.classifier[1].in_features
                    self.nnet.classifier = torch.nn.Sequential(nn.Dropout(p=0.2),
                                                               nn.Linear(in_features=num_fs, out_features=640,
                                                                         bias=True),
                                                               nn.Dropout(p=0.2),
                                                               nn.Linear(in_features=640, out_features=320, bias=True),
                                                               nn.Dropout(p=0.2),
                                                               nn.Linear(in_features=320, out_features=1, bias=True),
                                                               )


                elif 'map' in self.model_type:
                    self.nnet.features[-1][0] = nn.Conv2d(320, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
                    self.nnet.features[-1][1] = nn.BatchNorm2d(640, eps=1e-05, momentum=0.1, affine=True,
                                                               track_running_stats=True)
                    self.nnet.features.append(nn.Conv2d(640, 240, kernel_size=(1, 1), stride=(1, 1), bias=False))
                    self.nnet.features.append(
                        nn.BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                    self.nnet.features.append(nn.Conv2d(240, 64, kernel_size=(1, 1), stride=(1, 1), bias=False))
                    self.nnet.features.append(
                        nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                    self.nnet.features.append(nn.Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False))
                    self.nnet.features.append(
                        nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                    self.nnet.features.append(nn.Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1), bias=True))
                    self.nnet.avgpool = torch.nn.Identity()

                    self.nnet.classifier = torch.nn.Sequential()
                    if self.config.add_activ_14reso:  # for visualize, do not loss descend
                        # ipdb.set_trace()
                        self.nnet = Effnet_2outputs_AGB_activMap(self.nnet, arch='eff')


        elif 'torchEfficientnetb2' in self.model_type:
            if self.config.task == 'classification':
                self.nnet = models.efficientnet_b2(pretrained=self.pretrained)
                num_fs = self.nnet.classifier[1].in_features
                self.nnet.classifier = torch.nn.Sequential(nn.Dropout(p=0.2, inplace=True),
                                                           nn.Linear(in_features=num_fs, out_features=640, bias=True),
                                                           nn.Dropout(p=0.2, inplace=True),
                                                           nn.Linear(in_features=640, out_features=320, bias=True),
                                                           nn.Dropout(p=0.2, inplace=True),
                                                           nn.Linear(in_features=320, out_features=64, bias=True),
                                                           nn.Dropout(p=0.2, inplace=True),
                                                           nn.Linear(in_features=64, out_features=1, bias=True))

            elif self.config.task == 'regression':
                assert self.pretrained == 1
                print('model architecture loaded for ', self.model_type)
                self.nnet = models.efficientnet_b2(pretrained=self.pretrained)
                if 'dense' in self.model_type:
                    num_fs = self.nnet.classifier[1].in_features
                    self.nnet.classifier = torch.nn.Sequential(nn.Dropout(p=0.2),
                                                               nn.Linear(in_features=num_fs, out_features=640,
                                                                         bias=True),
                                                               nn.Dropout(p=0.2),
                                                               nn.Linear(in_features=640, out_features=320, bias=True),
                                                               nn.Dropout(p=0.2),
                                                               nn.Linear(in_features=320, out_features=64, bias=True),
                                                               nn.Dropout(p=0.2),
                                                               nn.Linear(in_features=64, out_features=1, bias=True))

                elif 'map' in self.model_type:
                    # ipdb.set_trace()
                    num_fs = self.nnet.features[-2][-1].block[-1][0].out_channels
                    self.nnet.features[-1][0] = nn.Conv2d(num_fs, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
                    self.nnet.features[-1][1] = nn.BatchNorm2d(640, eps=1e-05, momentum=0.1, affine=True,
                                                               track_running_stats=True)
                    self.nnet.features.append(nn.Conv2d(640, 240, kernel_size=(1, 1), stride=(1, 1), bias=False))
                    self.nnet.features.append(
                        nn.BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                    self.nnet.features.append(nn.Conv2d(240, 64, kernel_size=(1, 1), stride=(1, 1), bias=False))
                    self.nnet.features.append(
                        nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                    self.nnet.features.append(nn.Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False))
                    self.nnet.features.append(
                        nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                    self.nnet.features.append(nn.Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1), bias=True))
                    self.nnet.avgpool = torch.nn.Identity()

                    self.nnet.classifier = torch.nn.Sequential()
                    if self.config.add_activ_14reso:  # for visualize, do not loss descend
                        # ipdb.set_trace()
                        self.nnet = Effnet_2outputs_AGB_activMap(self.nnet, arch='eff')

        elif self.model_type == 'torchEfficientnet_v2_s':
            assert self.pretrained == 1
            self.nnet = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
            if self.keep_all_features:
                num_fs = self.nnet.classifier[1].in_features
                self.nnet.classifier = torch.nn.Sequential(nn.Dropout(p=0.2),
                                                           nn.Linear(in_features=num_fs, out_features=640, bias=True),
                                                           nn.Dropout(p=0.2),
                                                           nn.Linear(in_features=640, out_features=320, bias=True),
                                                           nn.Dropout(p=0.2),
                                                           nn.Linear(in_features=320, out_features=1, bias=True))
            else:
                # remove two feature blocks, reduce parameters to 5M
                # import ipdb
                # ipdb.set_trace()
                num_fs = self.nnet.features[-2][0].block[0][0].in_channels  # 160
                # num_fs = self.nnet.features[-1][0].in_channels # 160
                # assert num_fs == 160
                self.nnet.features = nn.Sequential(*list(self.nnet.features.children())[:-2])
                self.nnet.classifier = torch.nn.Sequential(nn.Dropout(p=0.4),
                                                           nn.Linear(in_features=num_fs, out_features=80, bias=True),
                                                           nn.Dropout(p=0.4),
                                                           nn.Linear(in_features=80, out_features=20, bias=True),
                                                           nn.Dropout(p=0.4),
                                                           nn.Linear(in_features=20, out_features=1, bias=True))

        elif self.model_type == 'resnet18':
            self.nnet = models.resnet18(pretrained=self.pretrained, num_classes=self.output_ch)
            self.nnet.conv1 = nn.Conv2d(3, 64, (7, 7), (1, 1), (3, 3), bias=False)




        elif 'regnetY800MF' in self.model_type:
            assert self.pretrained == 1
            self.nnet = models.regnet_y_800mf(weights=models.RegNet_Y_800MF_Weights.IMAGENET1K_V2)
            # ipdb.set_trace()
            num_fs = self.nnet.trunk_output[-1][-1].f.c[0].out_channels  # 748
            if 'dense' in self.model_type:
                self.nnet.fc = torch.nn.Sequential(nn.Dropout(p=0.2),
                                                   # numfs close to 640
                                                   nn.Linear(in_features=num_fs, out_features=320, bias=True),
                                                   nn.Dropout(p=0.2),
                                                   nn.Linear(in_features=320, out_features=64, bias=True),
                                                   nn.Dropout(p=0.2),
                                                   nn.Linear(in_features=64, out_features=1, bias=True))

            elif 'map' in self.model_type:
                self.nnet.trunk_output.append(nn.Conv2d(num_fs, 240, kernel_size=(1, 1), stride=(1, 1), bias=False))
                self.nnet.trunk_output.append(
                    nn.BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                self.nnet.trunk_output.append(nn.Conv2d(240, 64, kernel_size=(1, 1), stride=(1, 1), bias=False))
                self.nnet.trunk_output.append(
                    nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                self.nnet.trunk_output.append(nn.Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False))
                self.nnet.trunk_output.append(
                    nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                self.nnet.trunk_output.append(nn.Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1), bias=True))
                self.nnet.avgpool = torch.nn.Identity()

                # self.nnet.fc = torch.nn.Sequential()
                self.nnet.fc = torch.nn.Identity()
                # ipdb.set_trace()
                if self.config.add_activ_loss_descend and not self.config.add_activ_flatten:
                    # ipdb.set_trace()
                    self.nnet = Effnet_2outputs_AGB_activMap(self.nnet, arch='reg')
                elif self.config.mode == 'AttentiveSelection':
                    ipdb.set_trace()
                    # freeze the base weights and fintune an attention map to weight more rational
                    self.nnet = Effnet_AGB_activMap_attention(self.nnet, 1)
                elif self.config.add_activ_14reso:  # for visualize, do not loss descend
                    # ipdb.set_trace()
                    self.nnet = Effnet_2outputs_AGB_activMap(self.nnet, arch='reg')


        elif 'mobilenetv3' in self.model_type:
            assert self.pretrained == 1
            self.nnet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
            # ipdb.set_trace()
            num_fs = self.nnet.features[-1][0].out_channels  # 960
            if 'dense' in self.model_type:
                self.nnet.classifier = torch.nn.Sequential(nn.Dropout(p=0.2),
                                                           nn.Linear(in_features=num_fs, out_features=640, bias=True),
                                                           nn.Dropout(p=0.2),
                                                           nn.Linear(in_features=640, out_features=320, bias=True),
                                                           nn.Dropout(p=0.2),
                                                           nn.Linear(in_features=320, out_features=64, bias=True),
                                                           nn.Dropout(p=0.2),
                                                           nn.Linear(in_features=64, out_features=1, bias=True))

            elif 'map' in self.model_type:
                self.nnet.features.append(nn.Conv2d(num_fs, 640, kernel_size=(1, 1), stride=(1, 1), bias=False))
                self.nnet.features.append(
                    nn.BatchNorm2d(640, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                self.nnet.features.append(nn.Conv2d(640, 240, kernel_size=(1, 1), stride=(1, 1), bias=False))
                self.nnet.features.append(
                    nn.BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                self.nnet.features.append(nn.Conv2d(240, 64, kernel_size=(1, 1), stride=(1, 1), bias=False))
                self.nnet.features.append(
                    nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                self.nnet.features.append(nn.Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False))
                self.nnet.features.append(
                    nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                self.nnet.features.append(nn.Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1), bias=True))
                self.nnet.avgpool = torch.nn.Identity()

                self.nnet.classifier = torch.nn.Sequential()

        if not self.pretrained:

            self.optimizer = optim.Adam(list(self.nnet.parameters()),
                                        self.lr, [self.beta1, self.beta2], self.weightDecay)

        else:  # using pretrained model
            params_to_update = self.nnet.parameters()
            print("Params to learn:")

            params_to_update = []
            for name, param in self.nnet.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t", name)

            # Observe that all parameters are being optimized
            self.optimizer = optim.Adam(params_to_update, self.lr, [self.beta1, self.beta2], self.weightDecay)

        self.nnet.to(self.device)

        self.print_network(self.nnet, self.model_type)
        # ipdb.set_trace()

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        # print(model)
        # import ipdb
        # ipdb.set_trace()

        print('------Model name------- ', name)

        model_stats = summary(model, input_size=[self.config.BATCH_SIZE, self.config.img_ch, self.config.inputlength, self.config.inputlength], col_names = ("input_size", "output_size", "num_params"))
        # model_stats = summary(model, input_size=(32, 3, 180, 180))
        print(model_stats)
        print("The number of parameters: {}".format(num_params))
        self.summary_str = str(model_stats)


    def load_weights(self):
        if os.path.isfile(self.config.model_path):

            if self.config.model_path.endswith('pt'):
                checkpoint = torch.load(self.config.model_path, map_location='cuda:0')
                # ipdb.set_trace()
                self.nnet.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.nnet.load_state_dict(torch.load(self.config.model_path))
            # # Load the pretrained Encoder
            # checkpoint = torch.load(self.config.model_path)
            # # ipdb.set_trace()
            # self.nnet.load_state_dict(checkpoint['model_state_dict'])
            print('%s is Successfully Loaded from %s'%(self.config.model_type,self.config.model_path))
        # self.unet.load_state_dict(torch.load(self.model_path))

        self.nnet.train(False)
        self.nnet.eval()


    def predict_all(self):
        c, notwork, nochm = predict_AGB(self.config, self.all_files, self.nnet, self.device)
        return c, notwork, nochm


def load_files(config):
    all_files0 = glob.glob(f"{config.input_image_dir}/{config.input_image_pref}*{config.input_image_type}", recursive=False)
    all_files = [(i, os.path.basename(i)) for i in all_files0]
    print('**********************************')
    print('Number of raw image to predict:', len(all_files))
    return all_files


def addTOResult(res, prediction, row, col, he, wi, operator = 'MAX'):
    currValue = res[row:row+he, col:col+wi]
    newPredictions = prediction
    # ipdb.set_trace()
    try:
            # IMPORTANT: MIN can't be used as long as the mask is initialed with 0!!!!! If you want to use MIN initial the mask with -1 and handle the case of default value(-1) separately.
        if operator == 'MIN': # Takes the min of current prediction and new prediction for each pixel
            currValue [currValue == -1] = 1 #Replace -1 with 1 in case of MIN
            resultant = np.minimum(currValue, newPredictions)
        elif operator == 'MAX':
            resultant = np.maximum(currValue, newPredictions)
        elif operator == "MIX": # alpha blending # note do not combine with empty regions
            mm1 = currValue!=0
            currValue[mm1] = currValue[mm1] * 0.5 + newPredictions[mm1] * 0.5
            mm2 = (currValue==0)
            currValue[mm2] = newPredictions[mm2]
            resultant = currValue
        else: #operator == 'REPLACE':
            resultant = newPredictions
        res[row:row+he, col:col+wi] =  resultant
    except:
        # for with activation map
        # image boundary
        newPredictions = prediction[:currValue.shape[0], :currValue.shape[1]]
        if operator == 'MIN': # Takes the min of current prediction and new prediction for each pixel
            currValue [currValue == -1] = 1 #Replace -1 with 1 in case of MIN
            resultant = np.minimum(currValue, newPredictions)
        elif operator == 'MAX':
            resultant = np.maximum(currValue, newPredictions)
        elif operator == "MIX": # alpha blending # note do not combine with empty regions
            mm1 = currValue!=0
            currValue[mm1] = currValue[mm1] * 0.5 + newPredictions[mm1] * 0.5
            mm2 = (currValue==0)
            currValue[mm2] = newPredictions[mm2]
            resultant = currValue
        else: #operator == 'REPLACE':
            resultant = newPredictions
        res[row:row+he, col:col+wi] =  resultant
    return (res)



def predict_run(model, device, batch, batch_pos, mask, operator, upscale = 1, sum = 1):

    images = np.stack(batch, axis = 0) #stack a list of arrays along axis
    images = torch.from_numpy(images)
    images = images.to(device, dtype=torch.float)
    pred = model(images).squeeze()
    if sum:
        # pred = pred.cpu().detach().numpy().sum(axis = (1))
        pred = pred.cpu().detach().numpy().sum(axis=(1, 2))
    else:
        pred = pred.cpu().detach().numpy()
    for i in range(len(batch_pos)):
        (col, row, wi, he) = batch_pos[i]
        p = pred[i]
        # ipdb.set_trace()
        # for AGB, p is a single value
        mask = addTOResult(mask, p, row, col, he, wi, operator)
    return mask

def predict_run_2outputs(model, device, batch, batch_pos, batch_pos_act, mask, mask_act, operator, upscale = 1):

    images = np.stack(batch, axis = 0) #stack a list of arrays along axis
    images = torch.from_numpy(images)
    images = images.to(device, dtype=torch.float)
    pred = model(images).squeeze()
    # enfore non-negative
    pred = torch.clamp(pred, min=0)
    pred = pred.cpu().detach().numpy()

    pred_act = pred.reshape((len(batch), 14, 14))
    # set outer ring values to 0
    pred_act[:, 0, :] = 0
    pred_act[:, -1, :] = 0
    pred_act[:, :, 0] = 0
    pred_act[:, :, -1] = 0
    pred_AGB = pred.sum(axis = (-2,-1))

    for i in range(len(batch_pos)):
        (col, row, wi, he) = batch_pos[i]
        (col2, row2, wi2, he2) = batch_pos_act[i]
        p_agb = pred_AGB[i]
        p_act = pred_act[i]
        # if i == 62:
        #     ipdb.set_trace()
        # for AGB, p is a single value
        mask = addTOResult(mask, p_agb, row, col, he, wi, operator)
        mask_act = addTOResult(mask_act, p_act, row2, col2, he2, wi2, operator)
    #

    return mask, mask_act


def predict_AGB_img(config, model, device, paths, width=200, height=200, stride = 200, upscale = 1):

    if config.add_chm:
        imgPath, chmPath = paths
    else:
        imgPath = paths[0]

    with rasterio.open(imgPath) as img:
        nols, nrows = img.meta['width'], img.meta['height'] # 5000, 5000
        meta = img.meta.copy()

        if 'float' not in meta['dtype']:
            meta['dtype'] = np.float32

        offsets = product(range(0, nols, stride), range(0, nrows, stride))
        big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)

        # biomass mask (coarser resolution)
        nr_mask = int(nrows*upscale/height)
        nc_mask = int(nols*upscale/width)
        masks = np.zeros((nr_mask, nc_mask), dtype=np.float32)

        meta.update(
                    {'width': int(nols*upscale),
                     'height': int(nrows*upscale)
                    }
                    )

        # biomass actiovation mask in case of map models
        if config.output_activation:

            nr_mask_act = int(nrows*upscale/height*config.output_activation_dim)
            nc_mask_act = int(nols*upscale/width*config.output_activation_dim)
            masks_act = np.zeros((nr_mask_act, nc_mask_act), dtype=np.float32)


        # transform data
        Transform = []
        Transform.append(T.ToTensor())
        Transform.append(T.Resize((config.inputlength, config.inputlength)))

        if config.imageNetnorm:
            Transform.append(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))


        Transform = T.Compose(Transform)

        batch = []
        batch_pos = [ ]
        if config.output_activation:
            batch_pos_act = [ ]
        for col_off, row_off in tqdm(offsets):

            window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)

            if config.band_select:
                patch = np.zeros((int(len(config.new_bands)), config.inputlength, config.inputlength))
            else:
                patch = np.zeros((meta['count'], config.inputlength, config.inputlength))
            temp_im = img.read(
                                out_shape=(
                                img.count,
                                int(window.height*upscale),
                                int(window.width*upscale)
                            ),
                            resampling=Resampling.bilinear, window = window)
            # to filter out sand
            # ipdb.set_trace()
            temp_im = np.transpose(temp_im, axes=(1,2,0))
            maskred = temp_im[:,:,0]<200
            temp_im[~maskred,:] = 0

            if config.band_select: # using only subset of the bands
                temp_im = temp_im[:,:,config.new_bands]

            if config.add_chm:
                # add chm as 3rd band
                chmf = rasterio.open(chmPath) #chm layer
                window2 = windows.Window(window.col_off / 2, window.row_off / 2,
                                        int(window.width/2), int(window.height/2))

                chm = chmf.read(
                                    out_shape=(
                                    chmf.count,
                                    int(window2.height),
                                    int(window2.width)
                                ),
                                resampling=Resampling.bilinear, window = window2)

                chm = chm.astype(np.float32)
                # chm = np.transpose(chm, axes=(1,2,0)) # Channel at the end
                chm[chm<0]=0
                chm[chm>50]=50
                if config.chm_resample:
                    # usample chm
                    # chm = resize(chm, (int(chm.shape[0]*2), int(chm.shape[1]*2), 1), preserve_range=True)
                    # chm = chm * (chm.shape[0] / float(chm.shape[0]*2)) * (chm.shape[1] / float(chm.shape[1]*2))
                    # print('if channel last')
                    chm = zoom(chm, (1, 2,2), order=1)
                # when using height, convert to gray
                # ipdb.set_trace()
                temp_im = rgb2gray(temp_im)
                # expand gray
                temp_im = np.array([temp_im]*2).astype(np.float32)
                # combine color and chm
                # ipdb.set_trace()
                temp_im = np.concatenate((temp_im, chm), axis=0)

                temp_im = np.transpose(temp_im, axes=(1,2,0))
            # print(temp_im[:, :, 2].max())
            # ipdb.set_trace()
            # data transform
            temp_im = Transform(temp_im)
            # ipdb.set_trace()

            # print(window.height, window.width)
            patch[:config.inputlength, :config.inputlength] = temp_im
            batch.append(patch)
            batch_pos.append((int(window.col_off/height), int(window.row_off/width), 1, 1))
            if config.output_activation:
                batch_pos_act.append((int(window.col_off/height*config.output_activation_dim), int(window.row_off/width*config.output_activation_dim), config.output_activation_dim, config.output_activation_dim))
                # print(window.col_off)
                # print(batch_pos_act)
                # print(batch_pos)
            if (len(batch) == config.BATCH_SIZE):

                curmask = masks[:, :]
                if config.output_activation:
                    curmask_act = masks_act[:, :]
                    # ipdb.set_trace()
                    curmask, curmask_act = predict_run_2outputs(model, device, batch, batch_pos, batch_pos_act, curmask, curmask_act, config.operator, upscale = upscale)
                else:
                    curmask = predict_run(model, device, batch, batch_pos, curmask, config.operator, upscale = upscale, sum = config.sum_output)

                batch = []
                batch_pos = []
                batch_pos_act = []

        if batch:
            curmask = masks[:, :]
            if config.output_activation:
                curmask_act = masks_act[:, :]
                curmask, curmask_act = predict_run_2outputs(model, device, batch, batch_pos, batch_pos_act, curmask, curmask_act, config.operator, upscale = upscale)
            else:
                curmask = predict_run(model, device, batch, batch_pos, curmask, config.operator, upscale = upscale, sum = config.sum_output)

            batch = []
            batch_pos = []
            batch_pos_act = []
    if config.output_activation:
        return masks, masks_act, meta
    else:
        return masks, meta



def predict_AGB(config, all_files, model, device):
    counter = 1
    notwork = []
    nochm = []
    # in case no chm file exist, use 0s
    if config.add_chm:
        waterchm = config.input_chm_dir + 'CHM_640_59_TIF_UTM32-ETRS89/CHM_1km_6402_598.tif'
    outputFiles = []
    for fullPath, filename in tqdm(all_files):
        outputFile = os.path.join(config.output_dir, filename[:-4] + config.output_suffix + config.output_image_type)
        if config.output_activation:
            outputFile_act = outputFile.replace(config.output_suffix, config.output_act_suffix)

        if config.add_chm:
            # locate Chm file
            coor = fullPath[-12:-9]+ fullPath[-8:-5]
            chmbase = os.path.join(config.input_chm_dir, 'CHM_'+coor+'_TIF_UTM32-ETRS89/')
            chmPath = chmbase + os.path.basename(fullPath).replace(config.input_image_pref, config.input_chm_pref)
            if not os.path.exists(chmPath):
                nochm.append(fullPath)
                chmPath = waterchm
            paths = [fullPath, chmPath]
        else:
            paths = [fullPath]
        if not os.path.exists(outputFile):
            outputFiles.append(outputFile)
            # ipdb.set_trace()
            #try:

            if config.output_activation:
                detectedMask, detectedMask_act, detectedMeta = predict_AGB_img(config, model, device, paths,
                                                width = config.WIDTH, height = config.HEIGHT,
                                                stride = config.STRIDE, upscale = config.upscale,
                                                )
                writeMaskToDisk(detectedMask, detectedMeta, outputFile, image_type = config.output_image_type,
                                write_as_type = config.output_dtype, rescale=config.rescale_output)
                writeMaskToDisk(detectedMask_act, detectedMeta, outputFile_act, image_type = config.output_image_type,
                                write_as_type = config.output_dtype, rescale=config.rescale_output)


            else:
                detectedMask, detectedMeta = predict_AGB_img(config, model, device, paths,
                                                width = config.WIDTH, height = config.HEIGHT,
                                                stride = config.STRIDE, upscale = config.upscale,
                                                )
                writeMaskToDisk(detectedMask, detectedMeta, outputFile, image_type = config.output_image_type,
                                write_as_type = config.output_dtype, rescale=config.rescale_output)

            counter += 1

            # print('Skipping: Something went wrong!', fullPath)
            # ipdb.set_trace()
            # notwork.append(fullPath)
            # continue
        else:
            print('Skipping: File already analysed!', fullPath)

    return counter, notwork, nochm


def writeMaskToDisk(detected_mask, detected_meta, wp, image_type, write_as_type = 'uint8', rescale = 1):
    meta = detected_meta.copy()
    meta['dtype'] =  write_as_type
    meta['count'] = 1
    meta.update(
                        {'compress':'lzw',
                          'driver': 'GTiff',
                            'nodata': -1
                        }
                    )
    if rescale:
        # rescale to Mg/pixel not ha
        detected_mask = detected_mask*0.16
        # raw output unit: Mg/ha
    # make Positive
    detected_mask[detected_mask<0] = 0
    detected_mask = detected_mask.astype(write_as_type)
    assert detected_mask.ndim == 2
    with rasterio.open(wp, 'w', **meta) as outds:
        outds.write(detected_mask, 1)

    return



def rgb2gray(rgb):

    r, g, b = rgb[:, :,0], rgb[:, :,1], rgb[:, :,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray
