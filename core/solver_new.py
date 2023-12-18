import os
import numpy as np
import time
import datetime

import pandas as pd
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
import random
from core.evaluation import *
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error, precision_score, recall_score, accuracy_score, f1_score
import cv2
from torchinfo import summary
import wandb
import matplotlib.pyplot as plt  # plotting tools
import torchvision.models as models
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from utils.wandb_utils import Wandb
from utils.model_checkpoint import ModelCheckpoint
from scipy.optimize import curve_fit
from scipy.stats import linregress
from functools import partial
from torch import Tensor
import ipdb
from tifffile import imsave
import seaborn as sns
from matplotlib import rc,rcParams
from piqa import PSNR, SSIM
torch.autograd.set_detect_anomaly(True)
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False




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


class Solver(object):
    def __init__(self, config, train_loader, valid_loader, test_loader, patch_size, cfg_path, launch_wandb=1):
        torch.cuda.empty_cache()

        # Data loader
        self.config = config
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.year = config.year
        self.patch_size = patch_size
        self.conf_score = config.conf_score
        self.norm_wei = config.norm_wei
        # Models
        self.nnet = None
        self.pretrained = config.pretrained
        self.keep_all_features = config.keep_all_features
        self.optimizer = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch


        self.loss_name = config.loss_name
        self.augmentation_prob = config.augmentation_prob
        self.nnlevel = config.nnlevel
        # Hyper-parameters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.weightDecay = config.weightDecay

        # Training settings
        self.num_epochs = config.num_epochs
        # self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size
        self.suf = config.model_suf
        self.lr_decay_rate = config.lr_decay_rate
        self.lr_decay_frequency = config.lr_decay_frequency
        self.min_lr = config.min_lr
        self.add_input2 = config.add_input2
        self.add_outputs = config.add_outputs

        if self.conf_score: # adding sample weights to loss function
            reduc = 'none'
        else:
            reduc = 'mean'

        self.loss_func = config.loss_func
        if self.loss_func == 'L1':
            print('Loss function: L1')
            self.criterion = torch.nn.L1Loss(reduction=reduc)
        elif self.loss_func == 'L2':
            # try log loss
            print('Loss function: L2')
            self.criterion = torch.nn.MSELoss(reduction=reduc)
        elif self.loss_func == 'BCE':
            self.criterion = torch.nn.BCELoss(reduction=reduc)
        else:
            print('loss not specified')

        if self.add_outputs:
            self.loss_func_addop = config.loss_func_addop
            print('Loss function for added outputs: ', self.loss_func_addop)
            if self.loss_func_addop == 'L1':
                self.criterion_addop = torch.nn.L1Loss(reduction=reduc)
            elif self.loss_func_addop == 'L2':
                self.criterion_addop = torch.nn.MSELoss(reduction=reduc)
            elif self.loss_func_addop == 'BCElog':
                self.criterion_addop = torch.nn.BCEWithLogitsLoss(reduction=reduc)

        # Step size
        self.log_step = config.log_step
        self.val_step = config.val_step

        # Path
        self.model_path = config.model_path
        # self.result_path = config.result_path
        self.mode = config.mode
        self.checkpoint_dir = config.checkpoint_dir
        self.image_callback_freq = config.image_callback_freq


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = config.model_type
        self.t = config.t
        self.build_model()



        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        timestr = time.strftime("%Y%m%d-%H%M")
        if self.add_input2:
            ps = self.patch_size[0][-1]
        else:
            ps = self.patch_size[-1]

        if config.onlyRGB:
            if config.nir_rep_red:
                if not config.add_chmInput:
                    self.bandns = 'NIRGB'
                elif config.add_chmInput:
                    self.bandns = 'NIRGBCHM'

            else:
                if not config.add_chmInput:
                    self.bandns = 'RGB'
                elif config.add_chmInput:
                    self.bandns = 'RGBCHM'
        else:
            print('invalid band setting: check!')
        # ipdb.set_trace()
        if config.task == 'regression':

            if self.config.resume_train:
                self.nnet_path_dir = self.config.model_path
                # ipdb.set_trace()
            else:
                self.nnet_path_dir = os.path.join(self.model_path, 'AGB_%s-%s-Epo%d-yearAft_%d-patch_%d-Loss_%s-bands_%s-_%s' %(timestr, self.model_type,self.num_epochs,int(self.year), ps, self.loss_name, self.bandns, self.suf))

        elif config.task == 'classification':
            self.nnet_path_dir = os.path.join(self.model_path, 'classification_%s-%s-Epo%d-patch_%d-Loss_%s-bands_%s-_%s' %(timestr, self.model_type,self.num_epochs, ps, self.loss_name, self.bandns, self.suf))

        print('********************************************')
        print('Model path: ', self.nnet_path_dir)
        if self.mode == 'train' and not os.path.exists(self.nnet_path_dir):
            os.makedirs(self.nnet_path_dir)
# =============================================================================
        self.saveImages = config.saveImages
        if self.saveImages:
            self.image_callback_dir = os.path.join(config.imageCallbackDir, 'model_%s-%s'%(timestr, self.model_type))
            if not os.path.exists(self.image_callback_dir):
                os.makedirs(self.image_callback_dir)

        # adding wandb

        self._checkpoint: ModelCheckpoint = ModelCheckpoint(
        self.checkpoint_dir,
        self.config.model_type,
        self.config.mode,
        run_config=self.config,
        resume=0,
        )

        if launch_wandb:
            Wandb.launch(config, cfg_path, 1)
# =============================================================================





    def build_model(self):
        """Build generator and discriminator."""
        if 'torchEfficientnetb0' in self.model_type:
            if self.config.task == 'classification':
                raise NotImplementedError

            elif self.config.task == 'regression':
                if not self.pretrained:
                    self.nnet = models.efficientnet_b0(num_classes=self.output_ch)
                    self.nnet.features[0][0] = nn.Conv2d(self.img_ch, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    num_fs = self.nnet.classifier[1].in_features
                    self.nnet.classifier = torch.nn.Sequential(nn.Dropout(p=0.2),
                                                        nn.Linear(in_features=num_fs, out_features=640, bias=True),
                                                        nn.Dropout(p=0.2),
                                                        nn.Linear(in_features=640, out_features=320, bias=True),
                                                        nn.Dropout(p=0.2),
                                                        nn.Linear(in_features=320, out_features=1, bias=True),
                                                        )


                # generating a map instead
                else: # load pretrained, only reini the final layer
                    if 'dense' in self.model_type:
                        # load pretrained, only reini the final layer
                        # net0 = models.efficientnet_b0(pretrained=True)
                        net0 = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
                        num_fs = net0.classifier[1].in_features

                        if self.add_input2:
                            # add input 2
                            self.nnet = Effnet_2inputs(net0)


                        else:
                            # shrink regression output
                            self.nnet = net0
                            self.nnet.classifier = torch.nn.Sequential(nn.Dropout(p=0.2),
                                                                nn.Linear(in_features=num_fs, out_features=640, bias=True),
                                                                nn.Dropout(p=0.2),
                                                                nn.Linear(in_features=640, out_features=320, bias=True),
                                                                nn.Dropout(p=0.2),
                                                                nn.Linear(in_features=320, out_features=1, bias=True),
                                                                )




                    elif 'map' in self.model_type:
                        if 'nopadding' not in self.model_type:
                            net0 = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

                            # shrink regression output
                            self.nnet = net0
                            self.nnet.features[-1][0] = nn.Conv2d(320, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
                            self.nnet.features[-1][1] = nn.BatchNorm2d(640, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                            self.nnet.features.append(nn.Conv2d(640, 240, kernel_size=(1, 1), stride=(1, 1), bias=False))
                            self.nnet.features.append(nn.BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                            self.nnet.features.append(nn.Conv2d(240, 64, kernel_size=(1, 1), stride=(1, 1), bias=False))
                            self.nnet.features.append(nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                            self.nnet.features.append(nn.Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False))
                            self.nnet.features.append(nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                            self.nnet.features.append(nn.Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1), bias=True))
                            # self.nnet.features.append(nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                            # self.nnet.features[-1][1] = nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

                            # remove avg poly
                            self.nnet.avgpool = torch.nn.Identity()

                            self.nnet.classifier = torch.nn.Sequential()
                            # ipdb.set_trace()
                            if self.config.add_activ_14reso: # for visualize, do not loss descend
                                # ipdb.set_trace()
                                self.nnet = Effnet_2outputs_AGB_activMap(self.nnet, arch = 'eff')
                            # # remove early paddings, cant do it more later layers due to dim mismatch
                            # for module in self.nnet.features[:2].modules():
                            #     if isinstance(module, nn.Conv2d):
                            #         module.padding = (0, 0)

                        else: # with padding

                            net0 = _efficientnet(inverted_residual_setting, 0.2, last_channel = None, weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1,  progress = True)

                            # shrink regression output
                            self.nnet = net0
                            self.nnet.features[-4] = torch.nn.Sequential(nn.Conv2d(80, 64, kernel_size=(1, 1), stride=(1, 1), bias=False),
                                                        nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                            self.nnet.features[-3] = torch.nn.Sequential(nn.Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False),
                                                    nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                            self.nnet.features[-2] = torch.nn.Sequential(nn.Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1), bias=False))
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
                self.nnet.classifier = torch.nn.Sequential(nn.Dropout(p=0.2, inplace = True),
                                                           nn.Linear(in_features=num_fs, out_features=640, bias=True),
                                                           nn.Dropout(p=0.2, inplace = True),
                                                           nn.Linear(in_features=640, out_features=320, bias=True),
                                                           nn.Dropout(p=0.2, inplace = True),
                                                           nn.Linear(in_features=320, out_features=64, bias=True),
                                                           nn.Dropout(p=0.2, inplace = True),
                                                           nn.Linear(in_features=64, out_features=1, bias=True))

            elif self.config.task == 'regression':
                print('model architecture loaded for ', self.model_type)
                assert self.pretrained == 1
                self.nnet = models.efficientnet_b1(pretrained=self.pretrained)
                if 'dense' in self.model_type:
                    num_fs = self.nnet.classifier[1].in_features
                    self.nnet.classifier = torch.nn.Sequential(nn.Dropout(p=0.2),
                                                        nn.Linear(in_features=num_fs, out_features=640, bias=True),
                                                        nn.Dropout(p=0.2),
                                                        nn.Linear(in_features=640, out_features=320, bias=True),
                                                        nn.Dropout(p=0.2),
                                                        nn.Linear(in_features=320, out_features=1, bias=True),
                                                        )


                elif 'map' in self.model_type:
                    self.nnet.features[-1][0] = nn.Conv2d(320, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
                    self.nnet.features[-1][1] = nn.BatchNorm2d(640, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    self.nnet.features.append(nn.Conv2d(640, 240, kernel_size=(1, 1), stride=(1, 1), bias=False))
                    self.nnet.features.append(nn.BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                    self.nnet.features.append(nn.Conv2d(240, 64, kernel_size=(1, 1), stride=(1, 1), bias=False))
                    self.nnet.features.append(nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                    self.nnet.features.append(nn.Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False))
                    self.nnet.features.append(nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                    self.nnet.features.append(nn.Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1), bias=True))
                    self.nnet.avgpool = torch.nn.Identity()

                    self.nnet.classifier = torch.nn.Sequential()
                    if self.config.add_activ_14reso: # for visualize, do not loss descend
                        # ipdb.set_trace()
                        self.nnet = Effnet_2outputs_AGB_activMap(self.nnet, arch = 'eff')


        elif 'torchEfficientnetb2' in self.model_type:
            if self.config.task == 'classification':
                self.nnet = models.efficientnet_b2(pretrained=self.pretrained)
                num_fs = self.nnet.classifier[1].in_features
                self.nnet.classifier = torch.nn.Sequential(nn.Dropout(p=0.2, inplace = True),
                                                           nn.Linear(in_features=num_fs, out_features=640, bias=True),
                                                           nn.Dropout(p=0.2, inplace = True),
                                                           nn.Linear(in_features=640, out_features=320, bias=True),
                                                           nn.Dropout(p=0.2, inplace = True),
                                                           nn.Linear(in_features=320, out_features=64, bias=True),
                                                           nn.Dropout(p=0.2, inplace = True),
                                                           nn.Linear(in_features=64, out_features=1, bias=True))

            elif self.config.task == 'regression':
                assert self.pretrained == 1
                print('model architecture loaded for ', self.model_type)
                self.nnet = models.efficientnet_b2(pretrained=self.pretrained)
                if 'dense' in self.model_type:
                    num_fs = self.nnet.classifier[1].in_features
                    self.nnet.classifier = torch.nn.Sequential(nn.Dropout(p=0.2),
                                                        nn.Linear(in_features=num_fs, out_features=640, bias=True),
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
                    self.nnet.features[-1][1] = nn.BatchNorm2d(640, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    self.nnet.features.append(nn.Conv2d(640, 240, kernel_size=(1, 1), stride=(1, 1), bias=False))
                    self.nnet.features.append(nn.BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                    self.nnet.features.append(nn.Conv2d(240, 64, kernel_size=(1, 1), stride=(1, 1), bias=False))
                    self.nnet.features.append(nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                    self.nnet.features.append(nn.Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False))
                    self.nnet.features.append(nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                    self.nnet.features.append(nn.Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1), bias=True))
                    self.nnet.avgpool = torch.nn.Identity()

                    self.nnet.classifier = torch.nn.Sequential()
                    if self.config.add_activ_14reso: # for visualize, do not loss descend
                        # ipdb.set_trace()
                        self.nnet = Effnet_2outputs_AGB_activMap(self.nnet, arch = 'eff')

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
                num_fs = self.nnet.features[-2][0].block[0][0].in_channels # 160
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
            self.nnet = models.regnet_y_800mf(weights = models.RegNet_Y_800MF_Weights.IMAGENET1K_V2)
            # ipdb.set_trace()
            num_fs = self.nnet.trunk_output[-1][-1].f.c[0].out_channels # 748
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
                self.nnet.trunk_output.append(nn.BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                self.nnet.trunk_output.append(nn.Conv2d(240, 64, kernel_size=(1, 1), stride=(1, 1), bias=False))
                self.nnet.trunk_output.append(nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                self.nnet.trunk_output.append(nn.Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False))
                self.nnet.trunk_output.append(nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                self.nnet.trunk_output.append(nn.Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1), bias=True))
                self.nnet.avgpool = torch.nn.Identity()

                # self.nnet.fc = torch.nn.Sequential()
                self.nnet.fc = torch.nn.Identity()
                # ipdb.set_trace()
                if self.config.add_activ_loss_descend and not self.config.add_activ_flatten:
                    # ipdb.set_trace()
                    self.nnet = Effnet_2outputs_AGB_activMap(self.nnet, arch = 'reg')
                elif self.config.mode == 'AttentiveSelection':
                    ipdb.set_trace()
                    # freeze the base weights and fintune an attention map to weight more rational
                    self.nnet = Effnet_AGB_activMap_attention(self.nnet, 1)
                elif self.config.add_activ_14reso: # for visualize, do not loss descend
                    # ipdb.set_trace()
                    self.nnet = Effnet_2outputs_AGB_activMap(self.nnet, arch = 'reg')


        elif 'mobilenetv3' in self.model_type:
            assert self.pretrained == 1
            self.nnet = models.mobilenet_v3_large(weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
            # ipdb.set_trace()
            num_fs = self.nnet.features[-1][0].out_channels # 960
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
                self.nnet.features.append(nn.BatchNorm2d(640, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                self.nnet.features.append(nn.Conv2d(640, 240, kernel_size=(1, 1), stride=(1, 1), bias=False))
                self.nnet.features.append(nn.BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                self.nnet.features.append(nn.Conv2d(240, 64, kernel_size=(1, 1), stride=(1, 1), bias=False))
                self.nnet.features.append(nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                self.nnet.features.append(nn.Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False))
                self.nnet.features.append(nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                self.nnet.features.append(nn.Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1), bias=True))
                self.nnet.avgpool = torch.nn.Identity()

                self.nnet.classifier = torch.nn.Sequential()


        if not self.pretrained:

            self.optimizer = optim.Adam(list(self.nnet.parameters()),
                                      self.lr, [self.beta1, self.beta2], self.weightDecay)

        else: # using pretrained model
            params_to_update = self.nnet.parameters()
            print("Params to learn:")

            params_to_update = []
            for name,param in self.nnet.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t",name)


            # Observe that all parameters are being optimized
            self.optimizer = optim.Adam(params_to_update, self.lr, [self.beta1, self.beta2], self.weightDecay)



        self.nnet.to(self.device)

        self.print_network(self.nnet, self.model_type)
        # ipdb.set_trace()



    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            if p.requires_grad == True:
                num_params += p.numel()
        # print(model)
        # import ipdb
        # ipdb.set_trace()

        print('------Model name------- ', name)
        if self.add_input2:
            model_stats = summary(model, input_size=[self.patch_size[0], self.patch_size[1]], col_names = ("input_size", "output_size", "num_params"))
        else:
            model_stats = summary(model, input_size=self.patch_size, col_names = ("input_size", "output_size", "num_params"))
        # model_stats = summary(model, input_size=(32, 3, 180, 180))
        print(model_stats)
        print("The number of trainable parameters: {}".format(num_params))
        self.summary_str = str(model_stats)





    def train_epoch(self, epoch, lr):
        self.nnet.train(True)
        self.stage = 'train'
        batch_time = AverageMeter()
        losses = AverageMeter()
        if self.config.task == 'classification':
            f1 = AverageMeter()
            acc = AverageMeter()
            sens = AverageMeter()
            prec = AverageMeter()
        elif self.config.task == 'regression':
            maes = AverageMeter() # check normal loss without weights
            rmse = AverageMeter()
            mape = AverageMeter()
            r2 = AverageMeter()
        if self.add_outputs:
            loss_ad1 = AverageMeter()
            loss_ad2 = AverageMeter()
            loss_ad3 = AverageMeter()

        if self.config.add_activ_loss_descend:
            loss_activ = AverageMeter()

        end = time.time()
        st = time.time()
        for i, data in enumerate(self.train_loader):

            if self.conf_score:
                if self.add_input2:
                    images, input2, GT, sp_wei = data
                elif self.add_outputs:
                    images, GT, add_op1, add_op2, sp_wei = data
                elif self.config.add_activ_loss_descend or self.config.add_activ_14reso:
                    # ipdb.set_trace()
                    images, GT, chm_activ, sp_wei = data
                else:
                    images, GT, sp_wei = data
            else:
                images, GT = data
            # GT : Ground Truth

            images = images.to(self.device)
            GT = GT.to(self.device)

            if self.conf_score:
                sp_wei = sp_wei.to(self.device)
            if self.add_input2:
                input2 = input2.to(self.device)
            if self.add_outputs:
                add_op1, add_op2 = add_op1.to(self.device), add_op2.to(self.device)

            if self.add_input2:
                pred = self.nnet(images, input2).squeeze()
            else:
                if self.add_outputs:
                    pred = self.nnet(images)
                else:
                    pred = self.nnet(images).squeeze()

            if self.add_outputs:
                pred1, pred2, pred3 = pred
                pred1, pred2, pred3 = pred1.squeeze(), pred2.squeeze(), pred3.squeeze()

                loss1 = self.criterion(pred1,GT) # AGB loss
                loss2 = self.criterion_addop(pred2, add_op1)
                loss3 = self.criterion_addop(pred3, add_op2)
                loss = loss1 + loss2 + loss3 # weighted loss for 3 outputs
            else:   # one output
                if 'map' in self.model_type:

                    if self.config.add_activ_round_clip_pred:
                        # clip corners to remove artificts
                        im_length = images.shape[-1]
                        rad = int(im_length/2)
                        mask0 = create_circular_mask(rad*2, rad*2, radius=rad).astype(np.uint8)
                        mask0 = cv2.resize(mask0, (14, 14))
                        if self.config.add_activ_round_clip_inner_pred:
                            # clip more
                            mask00 = np.zeros((16, 16))
                            mask00[1:-1, 1:-1] = mask0
                            mask0 = cv2.resize(mask00, (14, 14))
                        mask = mask0 == 1
                        pred[:, ~mask] = 0

                    if self.config.add_activ_enforce_nonnegative:
                        pred = torch.clamp(pred, min = 0)

                    if self.config.add_activ_loss_descend or self.config.add_activ_14reso:
                        pred_activ = pred # activation map
                        activ_min = pred_activ.cpu().detach().numpy().min()
                        activ_max = pred_activ.cpu().detach().numpy().max()
                        if not self.config.add_activ_flatten:
                            # ipdb.set_trace()
                            pred = torch.sum(pred, (-2, -1))
                            if self.config.attri_label == 'h_can':
                                pred = pred/104
                        # need to sum first to get agb single value
                        # ipdb.set_trace()
                        else:

                            pred = torch.sum(pred, (-1))
                    elif self.mode == 'AttentiveSelection':
                        pred = torch.sum(pred.squeeze(), (-2, -1))
                    else:
                        pred = torch.sum(pred.squeeze(), (-1))

                if self.config.task == 'classification':
                    pred = torch.sigmoid(pred)
                # ipdb.set_trace()
                GT = GT.to(torch.float32)
                loss = self.criterion(pred,GT)

            if self.config.add_activ_loss_descend or self.config.add_activ_14reso:
                # loss compute also for activation map
                if self.config.add_activ_round_clip_pred:
                    # ipdb.set_trace()
                    chm_activ[:, :, ~mask] = 0
                chm_activ = chm_activ.to(self.device)
                if self.config.add_activ_flatten:
                    chm_activ = torch.flatten(chm_activ, start_dim=1) # 32, 7, 7, to 32, 49
                    # 0-1 normalization sample wise
                    chm_activ -= chm_activ.min(1, keepdim=True)[0]
                    chm_activ /= chm_activ.max(1, keepdim=True)[0]
                    pred_activ -= pred_activ.min(1, keepdim=True)[0]
                    pred_activ /= pred_activ.max(1, keepdim=True)[0]
                    chm_activ = torch.nan_to_num(chm_activ)
                    pred_activ = torch.nan_to_num(pred_activ)
                else:
                    # 0-1 normalization sample wise
                    # ipdb.set_trace()
                    chm_activ_raw = chm_activ.squeeze()
                    chm_activ = chm_activ_raw - torch.flatten(chm_activ_raw, start_dim=1).min(1, keepdim=True)[0].unsqueeze(-1)
                    chm_activ /= torch.flatten(chm_activ, start_dim=1).max(1, keepdim=True)[0].unsqueeze(-1)
                    pred_activ_raw = pred_activ.squeeze()
                    pred_activ = pred_activ_raw - torch.flatten(pred_activ_raw, start_dim=1).min(1, keepdim=True)[0].unsqueeze(-1)
                    pred_activ /= torch.flatten(pred_activ, start_dim=1).max(1, keepdim=True)[0].unsqueeze(-1)
                    chm_activ = torch.nan_to_num(chm_activ)
                    pred_activ = torch.nan_to_num(pred_activ)
                # ipdb.set_trace()
                if self.config.add_activ_loss_descend:
                    ssim = SSIM(n_channels=1, window_size=3).to(self.device)
                    # activ_loss = 1 - ssim(pred_activ.unsqueeze(1), chm_activ.unsqueeze(1))
                    activ_loss = - torch.log(ssim(pred_activ.unsqueeze(1), chm_activ.unsqueeze(1)))
                # ipdb.set_trace()
                # log images
                if i == 3 and self.config.log_activation_map:
                    # ipdb.set_trace()
                    randIndex = random.sample(range(self.config.batch_size), 8) # log 8 random samples
                    # ipdb.set_trace()
                    selec_refs=chm_activ_raw.cpu().detach().numpy()[randIndex, :,:]
                    selec_preds=pred_activ_raw.cpu().detach().numpy()[randIndex, :,:]
                    # ssims
                    # ipdb.set_trace()
                    # self.ssim_scores = [ssim(selec_preds.unsqueeze(1), selec_refs.unsqueeze(1))]
                    self.log_pair = np.concatenate((selec_refs, selec_preds), 1)#[..., np.newaxis]
                    # ipdb.set_trace()
                    selec_imgs = images.cpu().detach().numpy()[randIndex, :,:,:] # *(8, 3, 224, 224)
                    self.rawimages = [selec_imgs[i, :, :,:] for i in range(len(selec_imgs))]




            # adding sample weights to the loss
            # if True: loss(reduction = 'none)
            if self.conf_score:

                if self.norm_wei:
                    # # normlize loss weights
                    # loss =(loss * sp_wei / sp_wei.sum()).sum()
                    # loss = loss.mean()

                    if self.config.loss_weight_high: # weight high values more
                        loss = alpha_loss(loss, alpha = self.config.loss_alpha)

                    # do not take mean
                    loss =(loss * sp_wei).sum() / sp_wei.sum()

                    if self.add_outputs:
                        loss1 =(loss1 * sp_wei).sum() / sp_wei.sum()
                        loss2 =(loss2 * sp_wei).sum() / sp_wei.sum()
                        loss3 =(loss3 * sp_wei).sum() / sp_wei.sum()

                else:
                    loss = torch.mean(loss*sp_wei)

            # ipdb.set_trace()
            # losses.update(loss.item(), images.size(0))
            # losses.update(loss.item(), torch.count_nonzero(sp_wei).item())
            if self.config.add_activ_loss_descend:
                # merge two losses
                loss = loss + 10*activ_loss

            if self.conf_score:
                losses.update(loss.item(), sp_wei.sum().item())
            else:
                losses.update(loss.item(), images.size(0))
            # Backprop + optimize
            # self.reset_grad()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # after updating losses, calculate metrics
            if self.add_outputs:
                pred1 = pred1.cpu().detach().numpy()
                pred2 = pred2.cpu().detach().numpy()
                pred3 = pred3.cpu().detach().numpy()
                add_op1 = add_op1.cpu().detach().numpy()
                add_op2 = add_op2.cpu().detach().numpy()
            else:
                pred = pred.cpu().detach().numpy()
            GT = GT.cpu().detach().numpy()
            if self.conf_score:
                sp_wei = sp_wei.cpu().detach().numpy().mean() #check if mean or not
                # import ipdb
                # ipdb.set_trace()

            if self.add_outputs:
                pred = pred1 # loss is the combined loss, but other metrics are only for AGB

            if self.config.task == 'regression':
                # ipdb.set_trace()
                try:
                    rmse.update(mean_squared_error(GT, pred, squared=False).item(), images.size(0))
                    maes.update(mean_absolute_error(GT, pred).item(), images.size(0))
                    mape.update(mean_absolute_percentage_error(GT, pred).item(), images.size(0))
                    r2.update(r2_score(GT, pred).item(), images.size(0))
                except:
                    rmse.update(mean_squared_error(GT, pred.reshape(-1), squared=False).item(), images.size(0))
                    maes.update(mean_absolute_error(GT, pred.reshape(-1)).item(), images.size(0))
                    mape.update(mean_absolute_percentage_error(GT, pred.reshape(-1)).item(), images.size(0))
                    r2.update(r2_score(GT, pred.reshape(-1)), images.size(0))

            elif self.config.task == 'classification':
                # ipdb.set_trace()
                pred_lb = pred>0.5
                pred_lb = pred_lb.astype('int8')
                GT = GT.astype('int8')
                f1.update(f1_score(GT, pred_lb).item(), images.size(0))
                acc.update(accuracy_score(GT, pred_lb).item(), images.size(0))
                sens.update(recall_score(GT, pred_lb).item(), images.size(0))
                prec.update(precision_score(GT, pred_lb).item(), images.size(0))

            if self.add_outputs:
                # import ipdb
                # ipdb.set_trace()
                loss_ad1.update(loss1.item(), sp_wei.sum().item())
                loss_ad2.update(loss2.item(), sp_wei.sum().item())
                loss_ad3.update(loss3.item(), sp_wei.sum().item())

            if self.config.add_activ_loss_descend:
                loss_activ.update(activ_loss.item(), images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            # save image callback
            if self.saveImages:
                if i == 0 and epoch % self.image_callback_freq == 0:
                    images = images.cpu().detach().numpy()
                    self.image_callback(epoch, images, pred, GT, maes.avg)


        # Print the log info
        if self.conf_score:
            if self.add_outputs:
                print('Epoch [%d/%d] Epo time [%.2f] Itr time [%.2f] - Training, Loss: %.4f, MAE: %.4f, RMSE: %.4f, MAPE: %.4f, R^2: %.4f, sampleWeight: %.4f, Loss1: %.4f, Loss2: %.4f, Loss3: %.4f' %
                  (epoch+1, self.num_epochs, time.time() - st, batch_time.avg, \
                  losses.avg, maes.avg,\
                  rmse.avg, mape.avg ,r2.avg, sp_wei,
                  loss_ad1.avg, loss_ad2.avg, loss_ad3.avg))

            else:
                if self.config.add_activ_loss_descend:

                    print('Epoch [%d/%d] Epo time [%.2f] Itr time [%.2f] - Training, Loss: %.4f, activation loss: %.4f, MAE: %.4f, RMSE: %.4f, MAPE: %.4f, R^2: %.4f, sampleWeight: %.4f' %
                      (epoch+1, self.num_epochs, time.time() - st, batch_time.avg, \
                      losses.avg, loss_activ.avg, maes.avg,\
                      rmse.avg, mape.avg ,r2.avg, sp_wei))
                else:
                    print('Epoch [%d/%d] Epo time [%.2f] Itr time [%.2f] - Training, Loss: %.4f, MAE: %.4f, RMSE: %.4f, MAPE: %.4f, R^2: %.4f, sampleWeight: %.4f' %
                      (epoch+1, self.num_epochs, time.time() - st, batch_time.avg, \
                      losses.avg, maes.avg,\
                      rmse.avg, mape.avg ,r2.avg, sp_wei))
                    if 'map' in self.config.model_type:
                        print('activation map value range: ', activ_min, activ_max)
        else:
            if self.config.task == 'regression':
                print('Epoch [%d/%d] Epo time [%.2f] Itr time [%.2f] - Training, Loss/MAE: %.4f, RMSE: %.4f, MAPE: %.4f, R^2: %.4f' %
                      (epoch+1, self.num_epochs, time.time() - st, batch_time.avg, \
                      losses.avg,\
                      rmse.avg, mape.avg ,r2.avg))
            elif self.config.task == 'classification':
                print('Epoch [%d/%d] Epo time [%.2f] Itr time [%.2f] - Training, Loss/BCE: %.4f, acc: %.4f, F1: %.4f, Sens: %.4f, Prec: %.4f' %
                      (epoch+1, self.num_epochs, time.time() - st, batch_time.avg, \
                      losses.avg, acc.avg,\
                      f1.avg, sens.avg ,prec.avg))



        # Decay learning rate

        if epoch % self.lr_decay_frequency == 0:

            lr = self.lr * (self.lr_decay_rate ** (epoch // self.lr_decay_frequency))
            if lr > self.min_lr:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                print('Decay learning rate to {}.'.format(lr))
        # if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
        #     lr -= (self.lr / float(self.num_epochs_decay))
        #     for param_group in self.optimizer.param_groups:
        #         param_group['lr'] = lr
        #     print ('Decay learning rate to lr: {}.'.format(lr))


        if self.config.task == 'regression':
            self.finalize_epoch(epoch, losses.avg, rmse.avg, r2.avg)
        elif self.config.task == 'classification':
            self.finalize_epoch_classification(epoch, losses.avg, f1.avg, sens.avg)


        if self.conf_score:
            if self.add_outputs:
                del images, GT, sp_wei, add_op1, add_op2
            else:
                del images, GT, sp_wei
        else:
            del images, GT
        if self.config.task == 'regression':
            return losses.avg, rmse.avg, r2.avg
        elif self.config.task == 'classification':
            return losses.avg, f1.avg, sens.avg

    def train_epoch_Floss(self, epoch, lr):
        # loss update with forest branch
        self.nnet.train(True)
        self.stage = 'train'
        batch_time = AverageMeter()
        losses = AverageMeter() # total loss = forest loss + AGB losses
        maes = AverageMeter() # for AGB only
        rmse = AverageMeter() # for AGB only
        mape = AverageMeter() # for AGB only
        r2 = AverageMeter() # for AGB only
        if self.add_outputs:
            loss_ad1 = AverageMeter() # AGB loss
            loss_ad2 = AverageMeter() # forest loss
            loss_ad3 = AverageMeter() # b fraction loss


        end = time.time()
        st = time.time()
        for i, data in enumerate(self.train_loader):

            if self.conf_score:
                if self.add_input2:
                    images, input2, GT, sp_wei = data
                elif self.add_outputs:
                    images, GT, add_op1, add_op2, sp_wei = data # AGB, F, B
                else:
                    images, GT, sp_wei = data
            else:
                images, GT = data
            # GT : Ground Truth

            images = images.to(self.device)
            GT = GT.to(self.device)

            if self.conf_score:
                sp_wei = sp_wei.to(self.device)
            if self.add_input2:
                input2 = input2.to(self.device)
            if self.add_outputs:
                add_op1, add_op2 = add_op1.to(self.device), add_op2.to(self.device)

            if self.add_input2:
                pred = self.nnet(images, input2).squeeze()
            else:
                if self.add_outputs:
                    pred = self.nnet(images)
                else:
                    pred = self.nnet(images).squeeze()

            if self.add_outputs:
                pred_AGB, pred_F, pred_B = pred
                pred_AGB, pred_F, pred_B = pred_AGB.squeeze(), pred_F.squeeze(), pred_B.squeeze()

                loss1 = self.criterion(pred_AGB,GT) # AGB loss
                loss2 = self.criterion_addop(pred_F, add_op1) # F loss
                loss3 = self.criterion(pred_B, add_op2)
            else:   # one output
                loss = self.criterion(pred,GT)

            # adding sample weights to the loss
            # if True: loss(reduction = 'none)
            if self.conf_score:

                if self.norm_wei:
                    # # normlize loss weights
                    # loss =(loss * sp_wei / sp_wei.sum()).sum()
                    # loss = loss.mean()

                    # average before summing, since dims are different

                    loss1 =(loss1 * sp_wei).sum() / sp_wei.sum()
                    loss = loss1 + loss2 + loss3 # weighted  loss for 3 outputs
                    if self.add_outputs:
                        loss1 =(loss1 * sp_wei).sum() / sp_wei.sum()
                        loss2 =(loss2 * sp_wei).sum() / sp_wei.sum()
                        loss3 =(loss3 * sp_wei).sum() / sp_wei.sum()

                else:
                    loss = torch.mean(loss*sp_wei)


            # losses.update(loss.item(), images.size(0))
            # losses.update(loss.item(), torch.count_nonzero(sp_wei).item())
            if self.conf_score:
                losses.update(loss.item(), sp_wei.sum().item())
            else:
                losses.update(loss.item(), images.size(0))
            # Backprop + optimize
            # self.reset_grad()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # after updating losses, calculate metrics
            if self.add_outputs:
                pred1 = pred1.cpu().detach().numpy()
                pred2 = pred2.cpu().detach().numpy()
                pred3 = pred3.cpu().detach().numpy()
                add_op1 = add_op1.cpu().detach().numpy()
                add_op2 = add_op2.cpu().detach().numpy()
            else:
                pred = pred.cpu().detach().numpy()
            GT = GT.cpu().detach().numpy()
            if self.conf_score:
                sp_wei = sp_wei.cpu().detach().numpy().mean() #check if mean or not
                # import ipdb
                # ipdb.set_trace()

            if self.add_outputs:
                pred = pred1 # loss is the combined loss, but other metrics are only for AGB
            rmse.update(mean_squared_error(GT, pred, squared=False).item(), images.size(0))
            maes.update(mean_absolute_error(GT, pred).item(), images.size(0))
            mape.update(mean_absolute_percentage_error(GT, pred).item(), images.size(0))
            r2.update(r2_score(GT, pred).item(), images.size(0))

            if self.add_outputs:
                # import ipdb
                # ipdb.set_trace()
                loss_ad1.update(loss1.item(), sp_wei.sum().item())
                loss_ad2.update(loss2.item(), sp_wei.sum().item())
                loss_ad3.update(loss3.item(), sp_wei.sum().item())

            batch_time.update(time.time() - end)
            end = time.time()

            # save image callback
            if self.saveImages:
                if i == 0 and epoch % self.image_callback_freq == 0:
                    images = images.cpu().detach().numpy()
                    self.image_callback(epoch, images, pred, GT, maes.avg)


        # Print the log info
        if self.conf_score:
            if self.add_outputs:
                print('Epoch [%d/%d] Epo time [%.2f] Itr time [%.2f] - Training, Loss: %.4f, MAE: %.4f, RMSE: %.4f, MAPE: %.4f, R^2: %.4f, sampleWeight: %.4f, Loss1: %.4f, Loss2: %.4f, Loss3: %.4f' %
                  (epoch+1, self.num_epochs, time.time() - st, batch_time.avg, \
                  losses.avg, maes.avg,\
                  rmse.avg, mape.avg ,r2.avg, sp_wei,
                  loss_ad1.avg, loss_ad2.avg, loss_ad3.avg))

            else:
                print('Epoch [%d/%d] Epo time [%.2f] Itr time [%.2f] - Training, Loss: %.4f, MAE: %.4f, RMSE: %.4f, MAPE: %.4f, R^2: %.4f, sampleWeight: %.4f' %
                  (epoch+1, self.num_epochs, time.time() - st, batch_time.avg, \
                  losses.avg, maes.avg,\
                  rmse.avg, mape.avg ,r2.avg, sp_wei))
        else:
            print('Epoch [%d/%d] Epo time [%.2f] Itr time [%.2f] - Training, Loss/MAE: %.4f, RMSE: %.4f, MAPE: %.4f, R^2: %.4f' %
                  (epoch+1, self.num_epochs, time.time() - st, batch_time.avg, \
                  losses.avg,\
                  rmse.avg, mape.avg ,r2.avg))



        # Decay learning rate

        if epoch % self.lr_decay_frequency == 0:

            lr = self.lr * (self.lr_decay_rate ** (epoch // self.lr_decay_frequency))
            if lr > self.min_lr:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                print('Decay learning rate to {}.'.format(lr))
        # if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
        #     lr -= (self.lr / float(self.num_epochs_decay))
        #     for param_group in self.optimizer.param_groups:
        #         param_group['lr'] = lr
        #     print ('Decay learning rate to lr: {}.'.format(lr))

        if self.config.task == 'regression':
            self.finalize_epoch(epoch, losses.avg, rmse.avg, r2.avg)
        elif self.config.task == 'classification':
            self.finalize_epoch_classification(epoch, losses.avg, f1.avg, sens.avg)

        if self.conf_score:
            if self.add_outputs:
                del images, GT, sp_wei, add_op1, add_op2
            else:
                del images, GT, sp_wei
        else:
            del images, GT

        return losses.avg, rmse.avg, r2.avg



    def valid_epoch(self, epoch):
        st = time.time()
        self.nnet.train(False)
        self.nnet.eval()
        self.stage = 'val'
        batch_time = AverageMeter()
        val_losses = AverageMeter()

        if self.config.task == 'classification':
            val_f1 = AverageMeter()
            val_acc = AverageMeter()
            val_sens = AverageMeter()
            val_prec = AverageMeter()

        elif self.config.task == 'regression':
            val_rmse = AverageMeter()
            val_maes = AverageMeter() # check normal loss without weights
            val_mape = AverageMeter()


        if self.add_outputs:
            loss_ad1 = AverageMeter()
            loss_ad2 = AverageMeter()
            loss_ad3 = AverageMeter()
        val_r2 = AverageMeter()
        end = time.time()

        for i, data in enumerate(self.valid_loader):

            if self.conf_score:
                if self.add_input2:
                    images, input2, GT, sp_wei = data
                elif self.add_outputs:
                    images, GT, add_op1, add_op2, sp_wei = data
                elif self.config.add_activ_loss_descend or self.config.add_activ_14reso:
                    images, GT, chm_activ, sp_wei = data
                else:
                    images, GT, sp_wei = data

            else:
                images, GT = data
            images = images.to(self.device)
            GT = GT.to(self.device)

            if self.conf_score:
                sp_wei = sp_wei.to(self.device)

            if self.add_input2:
                input2 = input2.to(self.device)

            if self.add_outputs:
                add_op1, add_op2 = add_op1.to(self.device), add_op2.to(self.device)


            if self.add_input2:
                pred = self.nnet(images, input2).squeeze()
            else:
                if self.add_outputs:
                    pred = self.nnet(images)
                else:
                    pred = self.nnet(images).squeeze()



            if self.add_outputs:
                pred1, pred2, pred3 = pred
                pred1, pred2, pred3 = pred1.squeeze(), pred2.squeeze(), pred3.squeeze()
                loss1 = self.criterion(pred1,GT) # AGB loss
                loss2 = self.criterion_addop(pred2, add_op1)
                loss3 = self.criterion_addop(pred3, add_op2)
                loss = loss1 + loss2 + loss3 # weighted loss for 3 outputs
            else:   # one output
                if 'map' in self.model_type:
                    if self.config.add_activ_round_clip_pred:
                        # clip corners to remove artificts
                        im_length = images.shape[-1]
                        rad = int(im_length/2)
                        mask0 = create_circular_mask(rad*2, rad*2, radius=rad).astype(np.uint8)
                        mask0 = cv2.resize(mask0, (14, 14))
                        if self.config.add_activ_round_clip_inner_pred:
                            # clip more
                            mask00 = np.zeros((16, 16))
                            mask00[1:-1, 1:-1] = mask0
                            mask0 = cv2.resize(mask00, (14, 14))
                        mask = mask0==1
                        pred[:, ~mask] = 0

                    if self.config.add_activ_enforce_nonnegative:
                        pred = torch.clamp(pred, min = 0)

                    if self.config.add_activ_loss_descend or self.config.add_activ_14reso:
                        pred_activ = pred
                        activ_min = pred_activ.cpu().detach().numpy().min()
                        activ_max = pred_activ.cpu().detach().numpy().max()

                        if not self.config.add_activ_flatten:
                            pred = torch.sum(pred, (-2, -1))
                            if self.config.attri_label == 'h_can':
                                pred = pred/104
                        else:
                            pred = torch.sum(pred, (-1))
                    elif self.mode == 'AttentiveSelection':
                        pred = torch.sum(pred.squeeze(), (-2, -1))
                    else:
                        pred = torch.sum(pred.squeeze(), (-1))
                if self.config.task == 'classification':
                    pred = torch.sigmoid(pred)

                GT = GT.to(torch.float32)
                loss = self.criterion(pred,GT)
            # adding sample weights to the loss
            # if True: loss(reduction = 'none)

            if self.config.add_activ_loss_descend or self.config.add_activ_14reso:
                if self.config.add_activ_round_clip_pred:
                    chm_activ[:, :, ~mask] = 0
                # loss compute also for activation map
                chm_activ = chm_activ.to(self.device)
                if self.config.add_activ_flatten:
                    chm_activ = torch.flatten(chm_activ, start_dim=1) # 32, 7, 7, to 32, 49
                    # 0-1 normalization sample wise
                    chm_activ -= chm_activ.min(1, keepdim=True)[0]
                    chm_activ /= chm_activ.max(1, keepdim=True)[0]
                    pred_activ -= pred_activ.min(1, keepdim=True)[0]
                    pred_activ /= pred_activ.max(1, keepdim=True)[0]
                    chm_activ = torch.nan_to_num(chm_activ)
                    pred_activ = torch.nan_to_num(pred_activ)
                else:
                    # 0-1 normalization sample wise
                    # ipdb.set_trace()
                    chm_activ_raw = chm_activ.squeeze()
                    chm_activ = chm_activ_raw - torch.flatten(chm_activ_raw, start_dim=1).min(1, keepdim=True)[0].unsqueeze(-1)
                    chm_activ /= torch.flatten(chm_activ, start_dim=1).max(1, keepdim=True)[0].unsqueeze(-1)
                    pred_activ_raw = pred_activ.squeeze()
                    pred_activ = pred_activ_raw - torch.flatten(pred_activ_raw, start_dim=1).min(1, keepdim=True)[0].unsqueeze(-1)
                    pred_activ /= torch.flatten(pred_activ, start_dim=1).max(1, keepdim=True)[0].unsqueeze(-1)
                    chm_activ = torch.nan_to_num(chm_activ)
                    pred_activ = torch.nan_to_num(pred_activ)
                # ipdb.set_trace()
                if self.config.add_activ_loss_descend:
                    ssim = SSIM(n_channels=1, window_size=3).to(self.device)
                    # activ_loss = 1 - ssim(pred_activ.unsqueeze(1), chm_activ.unsqueeze(1))
                    activ_loss = - torch.log(ssim(pred_activ.unsqueeze(1), chm_activ.unsqueeze(1)))
                # ipdb.set_trace()
                # log images
                if i == 3 and self.config.log_activation_map:
                    # ipdb.set_trace()
                    randIndex = random.sample(range(self.config.batch_size), 8) # log 8 random samples
                    # ipdb.set_trace()
                    selec_refs=chm_activ_raw.cpu().detach().numpy()[randIndex, :,:]
                    selec_preds=pred_activ_raw.cpu().detach().numpy()[randIndex, :,:]
                    # ssims
                    # self.ssim_scores = [ssim(selec_preds.unsqueeze(1), selec_refs.unsqueeze(1))]
                    self.log_pair = np.concatenate((selec_refs, selec_preds), 1)#[..., np.newaxis]
                    # ipdb.set_trace()
                    selec_imgs = images.cpu().detach().numpy()[randIndex, :,:,:] # *(8, 3, 224, 224)
                    self.rawimages = [selec_imgs[i, :, :,:] for i in range(len(selec_imgs))]


            if self.config.conf_score_valid:
                if self.norm_wei:
                    # # normlize loss weights
                    # loss =(loss * sp_wei / sp_wei.sum()).sum()
                    # loss = loss.mean()
                    # do not take mean
                    loss =(loss * sp_wei).sum() / sp_wei.sum()
                    if self.add_outputs:
                        loss1 =(loss1 * sp_wei).sum() / sp_wei.sum()
                        loss2 =(loss2 * sp_wei).sum() / sp_wei.sum()
                        loss3 =(loss3 * sp_wei).sum() / sp_wei.sum()

                else:
                    loss = torch.mean(loss*sp_wei)
            else:
                loss = torch.mean(loss) # do not use weights for val set
            # val_losses.update(loss.item(), images.size(0))
            # val_losses.update(loss.item(), torch.count_nonzero(sp_wei).item())

            if self.config.add_activ_loss_descend:
                # merge two losses
                loss = loss + 10*activ_loss

            if self.config.conf_score_valid:
                val_losses.update(loss.item(), sp_wei.sum().item())
            else:
                val_losses.update(loss.item(), images.size(0))
            # Calculate Metrics #
            if self.add_outputs:
                pred1 = pred1.cpu().detach().numpy()
                pred2 = pred2.cpu().detach().numpy()
                pred3 = pred3.cpu().detach().numpy()
                add_op1 = add_op1.cpu().detach().numpy()
                add_op2 = add_op2.cpu().detach().numpy()
            else:
                pred = pred.cpu().detach().numpy()

            GT = GT.cpu().detach().numpy()
            if self.config.conf_score:
                sp_wei = sp_wei.cpu().detach().numpy().mean() #check if mean or not
            if self.add_outputs:
                pred = pred1 # loss is the combined loss, but other metrics are only for AGB
            if self.config.task == 'regression':
                try:
                    val_rmse.update(mean_squared_error(GT, pred, squared=False).item(), images.size(0))
                    val_maes.update(mean_absolute_error(GT, pred).item(), images.size(0))
                    val_mape.update(mean_absolute_percentage_error(GT, pred).item(), images.size(0))
                    val_r2.update(r2_score(GT, pred).item(), images.size(0))
                except:
                    val_rmse.update(mean_squared_error(GT, pred.reshape(-1), squared=False).item(), images.size(0))
                    val_maes.update(mean_absolute_error(GT, pred.reshape(-1)).item(), images.size(0))
                    val_mape.update(mean_absolute_percentage_error(GT, pred.reshape(-1)).item(), images.size(0))
                    val_r2.update(r2_score(GT, pred.reshape(-1)), images.size(0))
            elif self.config.task == 'classification':
                pred_lb = pred>0.5
                pred_lb = pred_lb.astype('int8')
                GT = GT.astype('int8')
                val_f1.update(f1_score(GT, pred_lb).item(), images.size(0))
                val_acc.update(accuracy_score(GT, pred_lb).item(), images.size(0))
                val_sens.update(recall_score(GT, pred_lb).item(), images.size(0))
                val_prec.update(precision_score(GT, pred_lb).item(), images.size(0))
            if self.add_outputs:
                loss_ad1.update(loss1.item(), sp_wei.sum().item())
                loss_ad2.update(loss2.item(), sp_wei.sum().item())
                loss_ad3.update(loss3.item(), sp_wei.sum().item())

            batch_time.update(time.time() - end)
            end = time.time()

            # save image callback
            if self.saveImages:
                if i == 0 and epoch % self.image_callback_freq == 0:
                    images = images.cpu().detach().numpy()
                    self.image_callback(epoch, images, pred, GT, val_maes.avg, stage = '_valid')

        if self.conf_score:
            if self.add_outputs:
                print('Epoch [%d/%d] Epo time [%.2f] Itr time [%.2f] - Validation, Loss: %.4f, MAE: %.4f, RMSE: %.4f, MAPE: %.4f, R^2: %.4f, sampleWeight: %.4f, Loss1: %.4f, Loss2: %.4f, Loss3: %.4f' %
                  (epoch+1, self.num_epochs, time.time() - st, batch_time.avg, \
                  val_losses.avg,val_maes.avg,\
                  val_rmse.avg, val_mape.avg, val_r2.avg, sp_wei,
                  loss_ad1.avg, loss_ad2.avg, loss_ad3.avg))
            else:
                print('Epoch [%d/%d] Epo time [%.2f] Itr time [%.2f] - Validation, Loss: %.4f, MAE: %.4f, RMSE: %.4f, MAPE: %.4f, R^2: %.4f, sampleWeight: %.4f' %
                  (epoch+1, self.num_epochs, time.time() - st, batch_time.avg, \
                  val_losses.avg,val_maes.avg,\
                  val_rmse.avg, val_mape.avg, val_r2.avg, sp_wei))
                if 'map' in self.config.model_type:
                    print('activation map value range: ', activ_min, activ_max)


        else:
            if self.config.task == 'regression':
                print('Epoch [%d/%d] Epo time [%.2f] Itr time [%.2f] - Validation, Loss/MAE: %.4f, RMSE: %.4f, MAPE: %.4f, R^2: %.4f' %
                      (epoch+1, self.num_epochs, time.time() - st, batch_time.avg, \
                      val_losses.avg,\
                      val_rmse.avg, val_mape.avg, val_r2.avg))
            elif self.config.task == 'classification':
                print('Epoch [%d/%d] Epo time [%.2f] Itr time [%.2f] - Training, Loss/BCE: %.4f, acc: %.4f, F1: %.4f, Sens: %.4f, Prec: %.4f' %
                      (epoch+1, self.num_epochs, time.time() - st, batch_time.avg, \
                      val_losses.avg, val_acc.avg,\
                      val_f1.avg, val_sens.avg ,val_prec.avg))

        if self.config.task == 'regression':
            self.finalize_epoch(epoch, val_losses.avg, val_rmse.avg, val_r2.avg)
        elif self.config.task == 'classification':
            self.finalize_epoch_classification(epoch, val_losses.avg, val_f1.avg, val_sens.avg)
        if self.config.conf_score:
            if self.add_outputs:
                del images, GT, sp_wei, add_op1, add_op2
            else:
                del images, GT, sp_wei
        else:
            del images, GT

        if self.config.task == 'regression':
            return val_losses.avg, val_rmse.avg, val_r2.avg
        elif self.config.task == 'classification':
            return val_losses.avg, val_f1.avg, val_sens.avg



    def image_callback(self, epoch, img, pred, lab, mae, stage = '_train'):
        plt.figure(figsize = (20,20)) # size(width, height)
        for i in range(4):
            for j in range(4): # column
                plt.subplot(4, 4, 4*j + i+1)
                curim = img[4*j + i]
                curim = np.transpose(curim, axes=(1,2,0)) # Channel at the end

                # recover from meanstd
                imstd = np.array([0.229, 0.224, 0.225])
                immean = np.array([0.485, 0.456, 0.406])
                curim = curim*imstd + immean
                plt.imshow(curim[:, :, :3])
                plt.title("label: %.1f;\n pred: %.1f"%(lab[4*j + i], pred[4*j + i]))

        plt.tight_layout()
        figpath = os.path.join(self.image_callback_dir, 'Epoch' + str(epoch) + stage + '_MAE_' + str(round(mae)) + '.jpg')
        plt.savefig(figpath, quality = 30)
        plt.clf()
        plt.close('all')








    def finalize_epoch(self, epoch, losses, rmse, r2):
        met_names = ['loss', 'rmse', 'r2']
        values = [losses, rmse, r2]

        metr = {}
        for k in range(len(met_names)):
            metr[met_names[k]]=values[k]

        metr2 = get_metrics(metr, self.stage)

        metrics = {}
        metrics['epoch']=epoch
        metrics['stage']=self.stage

        metrics['current_metrics'] = metr2
        # import ipdb
        # ipdb.set_trace()
        wandb.log(metr2, step=epoch)
        # log images
        if self.config.log_activation_map:
            im_list_raw = [wandb.Image(np.transpose(self.rawimages[i], (1, 2, 0))) for i in range(len(self.rawimages))]
            wandb.log({"random examples (img) on " + self.stage: im_list_raw })
            # ipdb.set_trace()
            #
            self.log_pair[self.log_pair<0]=0
            # self.log_pair *= 10 # for 7*7 resolution
            # for 14*14 resolution
            self.log_pair[:, :14, :]*=10
            if self.config.attri_label == 'h_can':
                self.log_pair[:, 14:, :]*=10
            elif self.config.attri_label == 'BMag_ha':
                self.log_pair[:, 14:, :]*=100
            # ipdb.set_trace()
            # self.log_pair[] *= 10
            self.log_pair[self.log_pair>255]=255
            self.log_pair = self.log_pair.astype(np.uint8)
            imm = [Image.fromarray(self.log_pair[i, :, :]) for i in range(len(self.log_pair))]
            # ipdb.set_trace()
            im_list = [wandb.Image(imm[i], caption = "top: chm reference; bottom: pred", mode = 'L') for i in range(len(imm))]
            # images = wandb.Image(self.log_pair, caption="Top: Reference CHM, Bottom: Pred", mode = 'F')
            # ipdb.set_trace()
            wandb.log({"random examples on " + self.stage: im_list})



        # self.publish_to_tensorboard(metrics, step)



        wandb.config.update({"model_name": self.config.model_type})

    def finalize_epoch_classification(self, epoch, losses, f1, sens):
        met_names = ['loss', 'f1', 'sens']
        values = [losses, f1, sens]

        metr = {}
        for k in range(len(met_names)):
            metr[met_names[k]]=values[k]

        metr2 = get_metrics_classification(metr, self.stage)

        metrics = {}
        metrics['epoch']=epoch
        metrics['stage']=self.stage

        metrics['current_metrics'] = metr2
        # import ipdb
        # ipdb.set_trace()
        wandb.log(metr2, step=epoch)


        # self.publish_to_tensorboard(metrics, step)

        wandb.config.update({"model_name": self.config.model_type})


    def train(self):
        #====================================== Training ===========================================#
        #===========================================================================================#

        nnet_path_model = os.path.join(self.nnet_path_dir, 'bestLoss.pt')
        # load trained models
        if os.path.isfile(nnet_path_model):
            # Load the pretrained Encoder
            self.nnet.load_state_dict(torch.load(nnet_path_model)['model_state_dict'])
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print('%s Weights are Successfully Loaded from %s'%(self.model_type,nnet_path_model))
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

            # read saved model architecture file
            print('Model architecture loaded from %s'%(nnet_path_model.replace('pt', 'log')))
            with open(nnet_path_model.replace('pt', 'log'), 'r') as f:
                lines = f.read()
                print(lines)



        lr = self.lr
        best_score = 100000
        if self.config.task == 'regression':
            best_rmse = 10000
            best_r2 = -100
        elif self.config.task == 'classification':
            best_f1 = -100
            best_sens = -100
        if self.config.earlystop:
            early_stopper = EarlyStopper(patience=self.config.patience)
        for epoch in range(self.num_epochs):

            # if self.pretrained:
                # if epoch == 3:
                #     print("Params to learn:")
                #     params_to_update = []
                #     for name,param in self.nnet.named_parameters():
                #         param.requires_grad == True
                #         params_to_update.append(param)
                #         print("\t",name)

                    # Observe that all parameters are being optimized
                    # self.optimizer = optim.Adam(params_to_update, self.lr, [self.beta1, self.beta2], self.weightDecay)

            # pytorch_total_params = sum(p.numel() for p in self.nnet.parameters() if p.requires_grad)
            # print('trainable params', pytorch_total_params)

            losses, met1, met2 = self.train_epoch(epoch, lr)

            val_loss, val_met1, val_met2 = self.valid_epoch(epoch)

            if self.config.earlystop:
                if early_stopper.early_stop(val_loss):
                    break

            # Save Best model
            if val_loss < best_score:
                best_score = val_loss
                best_nnet = self.nnet.state_dict()
                optim_stat = self.optimizer.state_dict()
                print('==========================================================')
                print('Best %s model score: Loss = %.4f'%(self.model_type,best_score))
                torch.save({'model_state_dict': best_nnet,
                            'optimizer_state_dict': optim_stat,
                            'epoch': self.config.num_epochs,
                            'best_loss': best_score
                            },
                            nnet_path_model)
                if not os.path.isfile(nnet_path_model.replace('pt', 'log')):
                    with open(nnet_path_model.replace('pt', 'log'), 'w') as f:
                        f.write(self.summary_str)

            if self.config.task == 'regression':
                if val_met2 > best_r2:
                    best_r2 = val_met2
                    best_nnet = self.nnet.state_dict()
                    optim_stat = self.optimizer.state_dict()
                    print('==========================================================')
                    print('Best %s model score: R2 = %.4f'%(self.model_type,best_r2))
                    torch.save({'model_state_dict': best_nnet,
                                'optimizer_state_dict': optim_stat,
                                'epoch': self.config.num_epochs,
                                'best_r2': best_r2
                                },
                                nnet_path_model.replace('Loss.pt', 'R2.pt'))

            elif self.config.task == 'classification':
                if val_met1 > best_f1: # met1: f1 score
                    best_f1 = val_met1
                    best_nnet = self.nnet.state_dict()
                    optim_stat = self.optimizer.state_dict()
                    print('==========================================================')
                    print('Best %s model score: F1 = %.4f'%(self.model_type,best_f1))
                    torch.save({'model_state_dict': best_nnet,
                                'optimizer_state_dict': optim_stat,
                                'epoch': self.config.num_epochs,
                                'best_r2': best_f1
                                },
                                nnet_path_model.replace('Loss.pt', 'F1.pt'))

    def AttentiveSelection(self):
        # ipdb.set_trace()
        new_model_path = os.path.dirname(self.nnet_path_dir)[:-1]+'_AttentiveSelection/'
        nnet_path_model = os.path.join(new_model_path, 'bestLoss.pt')
        # ipdb.set_trace()
        if not os.path.exists(new_model_path):
            os.mkdir(new_model_path)
        if self.model_path.endswith('pt'):
            checkpoint = torch.load(self.model_path)
            # ipdb.set_trace()
            module_src = dict(checkpoint['model_state_dict']) # dict
            module_dest = dict(self.nnet.named_parameters())
            # ipdb.set_trace()
            copyParams(module_src, module_dest)

            # self.nnet.load_state_dict(checkpoint['model_state_dict'])
        else:
            print('todo')
            # self.nnet.load_state_dict(torch.load(self.model_path))

        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('%s Weights are Successfully Loaded from %s'%(self.model_type, self.model_path))
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        self.print_network(self.nnet, self.model_type)

        for name, param in self.nnet.named_parameters():
            if param.requires_grad:
                print('=== Trainable parameters === ', name)


        # freeze weights for base model
        lr = self.lr
        best_score = 100000
        if self.config.task == 'regression':
            best_rmse = 10000
            best_r2 = -100
        elif self.config.task == 'classification':
            best_f1 = -100
            best_sens = -100
        if self.config.earlystop:
            early_stopper = EarlyStopper(patience=self.config.patience)
        for epoch in range(self.num_epochs):

            losses, met1, met2 = self.train_epoch(epoch, lr)

            val_loss, val_met1, val_met2 = self.valid_epoch(epoch)

            if self.config.earlystop:
                if early_stopper.early_stop(val_loss):
                    break

            # Save Best model
            if val_loss < best_score:
                best_score = val_loss
                best_nnet = self.nnet.state_dict()
                optim_stat = self.optimizer.state_dict()
                print('==========================================================')
                print('Best %s model score: Loss = %.4f'%(self.model_type,best_score))
                torch.save({'model_state_dict': best_nnet,
                            'optimizer_state_dict': optim_stat,
                            'epoch': self.config.num_epochs,
                            'best_loss': best_score
                            },
                            nnet_path_model)
                if not os.path.isfile(nnet_path_model.replace('pt', 'log')):
                    with open(nnet_path_model.replace('pt', 'log'), 'w') as f:
                        f.write(self.summary_str)

            if self.config.task == 'regression':
                if val_met2 > best_r2:
                    best_r2 = val_met2
                    best_nnet = self.nnet.state_dict()
                    optim_stat = self.optimizer.state_dict()
                    print('==========================================================')
                    print('Best %s model score: R2 = %.4f'%(self.model_type,best_r2))
                    torch.save({'model_state_dict': best_nnet,
                                'optimizer_state_dict': optim_stat,
                                'epoch': self.config.num_epochs,
                                'best_r2': best_r2
                                },
                                nnet_path_model.replace('Loss.pt', 'R2.pt'))

            elif self.config.task == 'classification':
                if val_met1 > best_f1: # met1: f1 score
                    best_f1 = val_met1
                    best_nnet = self.nnet.state_dict()
                    optim_stat = self.optimizer.state_dict()
                    print('==========================================================')
                    print('Best %s model score: F1 = %.4f'%(self.model_type,best_f1))
                    torch.save({'model_state_dict': best_nnet,
                                'optimizer_state_dict': optim_stat,
                                'epoch': self.config.num_epochs,
                                'best_r2': best_f1
                                },
                                nnet_path_model.replace('Loss.pt', 'F1.pt'))





    def train_transfer(self):
        print('===================================== Resume training / transfer learning ====================================')
        nnet_path_model = self.model_path.replace('.pt', '_transferred.pt')
        if os.path.isfile(self.model_path):
            # Load the pretrained Encoder
            checkpoint = torch.load(self.model_path)
            self.nnet.load_state_dict(checkpoint['model_state_dict'])
            print('%s is Successfully Loaded from %s'%(self.model_type,self.model_path))
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('Optimizer is Successfully Loaded from %s'%(self.model_path))
            start_epo = checkpoint['epoch']
            print('Resume training from epoch ', start_epo)
        else:
            print('********  weight loading failed, check model path!  ********')

        self.nnet.train(True)

        lr = self.lr
        if self.config.task == 'regression':
            try:
                best_score = checkpoint['best_r2']
            except:
                print('check!!!!!!!!!!!!!!')
                print('should load R2 saved model instead')
                best_score = checkpoint['best_loss']
        best_score -= 0.1
        if self.config.earlystop:
            early_stopper = EarlyStopper(patience=self.config.patience)
        for epoch in range(start_epo, self.num_epochs):

            losses, met1, met2 = self.train_epoch(epoch, lr)

            val_loss, val_met1, val_met2 = self.valid_epoch(epoch)

            if self.config.earlystop:
                if early_stopper.early_stop(val_loss):
                    break



            if self.config.task == 'regression':
                if val_met2 > best_score:
                    best_score = val_met2
                    best_nnet = self.nnet.state_dict()
                    optim_stat = self.optimizer.state_dict()
                    print('==========================================================')
                    print('Best %s model score: R2 = %.4f'%(self.model_type,best_score))
                    torch.save({'model_state_dict': best_nnet,
                                'optimizer_state_dict': optim_stat,
                                'epoch': self.config.num_epochs,
                                'best_r2': best_score
                                },
                                nnet_path_model)

            elif self.config.task == 'classification':
                if val_met1 > best_score: # met1: f1 score
                    best_score = val_met1
                    best_nnet = self.nnet.state_dict()
                    optim_stat = self.optimizer.state_dict()
                    print('==========================================================')
                    print('Best %s model score: F1 = %.4f'%(self.model_type,best_f1))
                    torch.save({'model_state_dict': best_nnet,
                                'optimizer_state_dict': optim_stat,
                                'epoch': self.config.num_epochs,
                                'best_r2': best_score
                                },
                                nnet_path_model.replace('Loss.pt', 'F1.pt'))



    def bias_correction(self, train_val_loader, val_val_loader):
        # using the training set to find b* and val set to validate
        if os.path.isfile(self.model_path):
            # Load the pretrained Encoder
            if self.model_path.endswith('pt'):
                checkpoint = torch.load(self.model_path, map_location='cuda:0')
                # ipdb.set_trace()
                self.nnet.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.nnet.load_state_dict(torch.load(self.model_path, map_location='cuda:0'))
            print('%s is Successfully Loaded from %s'%(self.model_type,self.model_path))
        
        
        self.nnet.train(False)
        self.nnet.eval()

        val_losses = AverageMeter()
        val_rmse = AverageMeter()
        val_mape = AverageMeter()
        val_maes = AverageMeter() # check normal loss without weights

        lb_list = []
        pd_list = []
        # print('len of loader',len(train_val_loader))
        for i, data in enumerate(train_val_loader):

            if self.conf_score:
                # if self.config.test_vis:
                #     images, GT, sp_wei, filen = data
                if self.config.add_activ_loss_descend or self.config.add_activ_14reso:
                    images, GT, chm_activ, sp_wei, filen = data
                else:
                    # ipdb.set_trace()
                    images, GT, sp_wei, filen = data
                
            else:
                images, GT = data

            images = images.to(self.device)
            GT = GT.to(self.device)

            if self.conf_score:
                sp_wei = sp_wei.to(self.device)

            pred = self.nnet(images).squeeze()
            if 'map' in self.model_type:
                # ipdb.set_trace()
                if self.config.add_activ_round_clip_pred:
                    # clip corners to remove artificts
                    im_length = images.shape[-1]
                    rad = int(im_length/2)
                    mask0 = create_circular_mask(rad*2, rad*2, radius=rad).astype(np.uint8)
                    mask0 = cv2.resize(mask0, (14, 14))
                    if self.config.add_activ_round_clip_inner_pred:
                        # clip more
                        mask00 = np.zeros((16, 16))
                        mask00[1:-1, 1:-1] = mask0
                        mask0 = cv2.resize(mask00, (14, 14))
                    mask = mask0==1
                    pred[:, ~mask] = 0
                if self.config.add_activ_enforce_nonnegative:
                    pred = torch.clamp(pred, min = 0)

                if self.config.add_activ_loss_descend or self.config.add_activ_14reso:
                    pred_activ = pred
                    if not self.config.add_activ_flatten:
                        if self.config.attri_label == 'h_can':
                            pred = pred/104
                        pred = torch.sum(pred, (-2, -1))
                    else:
                        pred = torch.sum(pred, (-1))
                else:
                    
                    pred = torch.sum(pred, (-1))

            GT = GT.to(torch.float32)
            
            loss = self.criterion(pred,GT)
            if self.conf_score:
                loss = torch.mean(loss*sp_wei)
                # compute for only samples with weight=1
                val_losses.update(loss.item(), torch.count_nonzero(sp_wei).item())
            else:
                val_losses.update(loss.item(), len(GT))
            # Calculate Metrics #
            pred = pred.cpu().detach().numpy()
            GT = GT.cpu().detach().numpy()

            imagesnp = images.cpu().detach().numpy()

            ## remove very wrong answers

            ttl = []
            try:
                for tt in range(len(pred)):
                    if abs(pred[tt]-GT[tt])>300:
                        ttl.append(tt)

                pred = [e for i, e in enumerate(pred) if i not in ttl]
                GT = [e for i, e in enumerate(GT) if i not in ttl]

            except:
                # batch size of 1
                if not abs(pred-GT)>300:
                    pred = [pred.tolist()]
                    GT = [GT.tolist()]
                else:
                    pred = []
                    GT = []

            # ipdb.set_trace()
            try:
                val_rmse.update(mean_squared_error(GT, pred, squared=False).item(), len(GT))
                val_mape.update(mean_absolute_percentage_error(GT, pred).item(), len(GT))
                val_maes.update(mean_absolute_error(GT, pred).item(), len(GT))
            except:
                continue # in case of []
            # if len(GT) == 1:
            #     GT = GT[0] # one patch left in the batch
            # ipdb.set_trace()
            lb_list.extend(GT)
            # print(lb_list)
            pd_list.extend(pred)
        if 'map' in self.config.model_type:
            self.b0 = self.nnet.im_features[-1].bias.item() 
            
        elif 'dense' in self.config.model_type:
            try:
                self.b0 = self.nnet.classifier[-1].bias.item()
            except:
                self.b0 = self.nnet.fc[-1].bias.item()
            
        print('b0:::::', self.b0)
        ppd = np.array(pd_list)
        ggt = np.array(lb_list)
        self.b_star = (ggt.sum() - (ppd- self.b0).sum())/len(ggt)
        print('b star::::', self.b_star)

        print('bias correction on val set with %d images, Loss: %.4f, MAE: %.4f, RMSE: %.4f, MAPE: %.4f' %
              (val_losses.count, val_losses.avg, val_maes.avg,\
              val_rmse.avg, val_mape.avg))


        median_absolute_error(ppd, ggt)
        ate = (ggt-ppd).sum()
        rte = ate/ggt.sum()
        print('relative total error', rte*100, '%')


        print('after bias correction')
        ppd_b = ppd - self.b0 + self.b_star

        median_absolute_error(ppd_b, ggt)
        ate = (ggt-ppd_b).sum()
        rte = ate/ggt.sum()
        print('relative total error after bias correction', rte*100, '%')

        print('----------------------On the validation set after bias correct----------------------')

        lb_list = []
        pd_list = []

        for i, data in enumerate(val_val_loader):

            if self.conf_score:
                if self.config.add_activ_loss_descend or self.config.add_activ_14reso:
                    images, GT, chm_activ, sp_wei, filen = data
                else:
                    # ipdb.set_trace()
                    images, GT, sp_wei, filen = data
            else:
                images, GT = data

            images = images.to(self.device)
            GT = GT.to(self.device)

            if self.conf_score:
                sp_wei = sp_wei.to(self.device)

            pred = self.nnet(images).squeeze()
            if 'map' in self.model_type:
                # ipdb.set_trace()
                if self.config.add_activ_round_clip_pred:
                    # clip corners to remove artificts
                    im_length = images.shape[-1]
                    rad = int(im_length/2)
                    mask0 = create_circular_mask(rad*2, rad*2, radius=rad).astype(np.uint8)
                    mask0 = cv2.resize(mask0, (14, 14))
                    if self.config.add_activ_round_clip_inner_pred:
                        # clip more
                        mask00 = np.zeros((16, 16))
                        mask00[1:-1, 1:-1] = mask0
                        mask0 = cv2.resize(mask00, (14, 14))
                    mask = mask0==1
                    pred[:, ~mask] = 0
                if self.config.add_activ_enforce_nonnegative:
                    pred = torch.clamp(pred, min = 0)

                if self.config.add_activ_loss_descend or self.config.add_activ_14reso:
                    pred_activ = pred
                    if not self.config.add_activ_flatten:
                        if self.config.attri_label == 'h_can':
                            pred = pred/104
                        pred = torch.sum(pred, (-2, -1))
                    else:
                        pred = torch.sum(pred, (-1))
                else:
                    
                    pred = torch.sum(pred, (-1))

            GT = GT.to(torch.float32)
            loss = self.criterion(pred,GT)
            if self.conf_score:
                loss = torch.mean(loss*sp_wei)


            # Calculate Metrics #
            pred = pred.cpu().detach().numpy()
            GT = GT.cpu().detach().numpy()
            imagesnp = images.cpu().detach().numpy()

            # ## remove very wrong answers
            # ttl = []
            # for tt in range(len(pred)):
            #     if abs(pred[tt]-GT[tt])>300:
            #         ttl.append(tt)
            #
            # pred = [e for i, e in enumerate(pred) if i not in ttl]
            # GT = [e for i, e in enumerate(GT) if i not in ttl]
            #
            try:

                pd_list.extend(pred)
                lb_list.extend(GT)

            except:
                pd_list.extend([pred.tolist()])
                lb_list.extend(GT)

        ppd = np.array(pd_list)
        ggt = np.array(lb_list)
        # ipdb.set_trace()
        median_absolute_error(ppd, ggt)
        ate = (ggt-ppd).sum()
        rte = ate/ggt.sum()
        print('relative total error', rte*100, '%')


        print('after bias correction')
        ppd_b = ppd - self.b0 + self.b_star

        median_absolute_error(ppd_b, ggt)
        ate = (ggt-ppd_b).sum()
        rte = ate/ggt.sum()
        print('relative total error after bias correction', rte*100, '%')

        return


    def test(self):
        #===================================== Test ====================================#
        # after all training epochs
        # self.build_model()
        if os.path.isfile(self.model_path):
            # Load the pretrained Encoder
            if self.model_path.endswith('pt'):
                checkpoint = torch.load(self.model_path, map_location='cuda:0')
                # ipdb.set_trace()
                self.nnet.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.nnet.load_state_dict(torch.load(self.model_path, map_location='cuda:0'))
            print('%s is Successfully Loaded from %s'%(self.model_type,self.model_path))

        self.nnet.train(False)
        self.nnet.eval()

        lb_list = []
        pd_list = []
        images_list = []
        # the old model way
        # if self.config.showmap:
        #     def get_features(name):
        #         def hook(model, input, output):
        #             features[name] = output.detach()
        #         return hook
        #
        #     try:
        #         self.nnet.features[-1].register_forward_hook(get_features('feats'))
        #     except:
        #         self.nnet.trunk_output[-1].register_forward_hook(get_features('feats'))
        #
        #
        #     FEATS = []
        #
        #     # placeholder for batch features
        #     features = {}

        if self.config.test_vis:
            filen_list = []
        for i, data in enumerate(self.test_loader):

            if self.conf_score:
                # if self.config.test_vis:
                #     images, GT, sp_wei, filen = data
                if self.config.add_activ_loss_descend or self.config.add_activ_14reso:
                    images, GT, chm_activ, sp_wei, filen = data
                else:
                    # ipdb.set_trace()
                    images, GT, sp_wei, filen = data
            else:
                images, GT = data

            images = images.to(self.device)
            GT = GT.to(self.device)
            # import ipdb
            # ipdb.set_trace()
            if self.conf_score:
                sp_wei = sp_wei.to(self.device)

            # ipdb.set_trace()
            pred = self.nnet(images).squeeze()
            # ipdb.set_trace()
            if 'map' in self.model_type:
                # ipdb.set_trace()
                if self.config.add_activ_round_clip_pred:
                    # clip corners to remove artificts
                    im_length = images.shape[-1]
                    rad = int(im_length/2)
                    mask0 = create_circular_mask(rad*2, rad*2, radius=rad).astype(np.uint8)
                    mask0 = cv2.resize(mask0, (14, 14))
                    if self.config.add_activ_round_clip_inner_pred:
                        # clip more
                        mask00 = np.zeros((16, 16))
                        mask00[1:-1, 1:-1] = mask0
                        mask0 = cv2.resize(mask00, (14, 14))
                    mask = mask0==1
                    pred[:, ~mask] = 0
                if self.config.add_activ_enforce_nonnegative:
                    pred = torch.clamp(pred, min = 0)

                if self.config.add_activ_loss_descend or self.config.add_activ_14reso:
                    pred_activ = pred
                    if not self.config.add_activ_flatten:
                        if self.config.attri_label == 'h_can':
                            pred = pred/104
                        pred = torch.sum(pred, (-2, -1))
                    else:
                        pred = torch.sum(pred, (-1))
                else:
                    
                    pred = torch.sum(pred, (-1))

            GT = GT.to(torch.float32)
            loss = self.criterion(pred,GT)
            if self.conf_score:
                loss = torch.mean(loss*sp_wei)

            # Calculate Metrics #
            pred = pred.cpu().detach().numpy()
            GT = GT.cpu().detach().numpy()
            imagesnp = images.cpu().detach().numpy()
            lb_list.extend(GT)
            try:
                pd_list.extend(pred)
            except:
                pd_list.extend([pred.tolist()])
            images_list.extend(imagesnp)
            if self.config.test_vis:
                filen_list.extend(filen)
            #
            for tt in range(len(imagesnp)):
                # ipdb.set_trace()
                if abs(GT[tt]-pred[tt]) > 200:

                    imstd = np.array([0.229, 0.224, 0.225])
                    immean = np.array([0.485, 0.456, 0.406])

                    curim = np.transpose(imagesnp[tt], axes=(1,2,0)) # Channel at the end
                    # print(curim.shape)
                    curim = curim*imstd + immean
                    plt.figure()
                    plt.imshow(curim)
                    bb = (pred[tt]-GT[tt])/(GT[tt] + 0.01)
                    plt.title("NFI label: %.1f;\n Prediction: %.1f;\n bias: %.2f"%(GT[tt], pred[tt], bb))
                    plt.show()
                    # plt.colorbar(imm, ax=ax[1])

            # for activation
            if self.config.showmap:
                # the old model activation layer
                # FEATS = features['feats'].cpu().numpy()
                #
                # print('- feats shape:', FEATS.shape)
                #
                # gray_scale = np.sum(FEATS, axis = (1))
                # gray_scale = gray_scale / FEATS.shape[1]
                # print(gray_scale.shape)
                # ipdb.set_trace()
                # pred_activ

                # ipdb.set_trace()
                activ_path = self.config.act_result_path + self.config.model_path.split('/AGB_')[1][:13]
                if not os.path.exists(activ_path):
                    os.mkdir(activ_path)

                pred_activ = pred_activ.cpu().detach().numpy()
                for idd in range(len(pred_activ)):

                    savename = os.path.join(activ_path, filen[idd] + '_act.tif')
                    # ipdb.set_trace()
                    imsave(savename,pred_activ[idd])


        # create a dataframe from filen_list, pd_list, lb_list
        df = pd.DataFrame({'filen':filen_list, 'pred':pd_list, 'label':lb_list})
        
        plot_resi(np.array(pd_list), np.array(lb_list))

        print('---------------------------On the test set Before bias correction-----------------------------')
        ppd = np.array(pd_list)
        ggt = np.array(lb_list)
        mdae = median_absolute_error(ggt, ppd)
        mae = mean_absolute_error(ggt, ppd)
        ate = (ggt-ppd).sum()
        rte = ate/ggt.sum()
        print('relative total error', rte*100, '%')
        print('median abs error', mdae)
        print('mae', mae)
        print('r2', r2_score(ggt, ppd))
        plot_scatter(ggt, ppd, 'Predicted biomass vs. target (unit = Mg C /ha)', 'Reference AGB (Mg/ha)',
             'Predicted AGB (Mg/ha)'
             ,550, font = 35, spinexy = 0, markersize = 30, xtic = 1, showr2 = 1, showfit = 0)

        plot_scatter(ggt, ppd, '', '',
          ''
          ,500, alpha = 0.7, font = 40, spinexy = 0, markersize = 50, xtic = 1, showr2 = 1, showfit = 0)


        if self.config.mode == 'bias_test':
            print('---------------------------on the test set After bias correction-----------------------------')
            ppd_b = ppd - self.b0 + self.b_star

            mdae = median_absolute_error( ggt, ppd_b)
            mae = mean_absolute_error(ggt, ppd_b)
            ate = (ggt-ppd_b).sum()
            rte = ate/ggt.sum()

            def MUPE(Y_actual,Y_Predicted): # underestimation
                mape = np.mean((Y_actual - Y_Predicted)/(Y_actual+1))*100
                return mape
            def smape(A, F):
                return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

            ggt_2 = ggt[ggt!=0]
            ppd_b_2 = ppd_b[ggt!=0]


            plot_scatter(ggt, ppd_b, 'Predicted biomass vs. target (unit = Mg C /ha)', 'Reference AGB (Mg/ha)',
                 'Predicted AGB (Mg/ha)'
                 ,550, font = 35, spinexy = 0, markersize = 30, xtic = 1, showr2 = 1, showfit = 0)

            plot_scatter(ggt, ppd_b, '', '',
              ''
              ,500, alpha = 0.7, font = 40, spinexy = 0, markersize = 50, xtic = 1, showr2 = 1, showfit = 0)


            print('smape %error after correction', smape(ggt_2,ppd_b_2))
            print('mape %error after correction', mean_absolute_percentage_error(ggt_2,ppd_b_2))
            # from sklearn.metrics import mean_absolute_percentage_error
            print('mean absolute error after correction', mae)
            print('rMAE after correction', mae/ggt.mean())
            print('root mean squared error after correction', mean_squared_error(ggt, ppd_b, squared=False))
            print('rRMSE after correction', mean_squared_error(ggt, ppd_b, squared=False)/ggt.mean())
            print('relative total error after bias correction', rte*100, '%')
            print('median abs error after correction', mdae)
            print('r2 after correction', r2_score(ggt, ppd_b))
            print('plot-based bias', ((ppd_b-ggt)/(ggt+0.01)).mean())





        if self.config.mode == 'bias_test':
            ppd_b, gtt, preds, gtts, intervals, gtm = load_data(ppd_b, ggt)
            
            if self.config.test_vis:
                return ppd_b, gtt, gtts, preds, mae, r2_score(ggt, ppd_b), mean_squared_error(ggt, ppd_b, squared=False), rte*100, filen_list
            else:

                return ppd_b, gtt, gtts, preds, mae, r2_score(ggt, ppd_b), mean_squared_error(ggt, ppd_b, squared=False), rte*100
        else:


            ppd, gtt, preds, gtts, intervals, gtm = load_data(ppd, ggt)
            # plot_box(ppd, gtt, gtts, preds,  font = 20, spinexy = 1, lim =350)
            # ipdb.set_trace()
            if self.config.test_vis:
                return ppd, gtt, gtts, preds, mae, r2_score(ggt, ppd), mean_squared_error(ggt, ppd, squared=False), rte*100, filen_list
            else:
                return ppd, gtt, gtts, preds, mae, r2_score(ggt, ppd), mean_squared_error(ggt, ppd, squared=False), rte*100


class EarlyStopper:
    def __init__(self, patience=50, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

import scipy.stats as stats
import scipy
# Modeling with Numpy
def equation(a, b):
    """Return a 1D polynomial."""
    return np.polyval(a, b)

def plot_ci_manual(t, s_err, n, x, x2, y2, ax=None):
    """Return an axes of confidence bands using a simple approach.

    Notes
    -----
    .. math:: \left| \: \hat{\mu}_{y|x0} - \mu_{y|x0} \: \right| \; \leq \; T_{n-2}^{.975} \; \hat{\sigma} \; \sqrt{\frac{1}{n}+\frac{(x_0-\bar{x})^2}{\sum_{i=1}^n{(x_i-\bar{x})^2}}}
    .. math:: \hat{\sigma} = \sqrt{\sum_{i=1}^n{\frac{(y_i-\hat{y})^2}{n-2}}}

    References
    ----------
    .. [1] M. Duarte.  "Curve fitting," Jupyter Notebook.
       http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/CurveFitting.ipynb

    """
    if ax is None:
        ax = plt.gca()

    ci = t * s_err * np.sqrt(1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
    ax.fill_between(x2, y2 + ci, y2 - ci, color="#b9cfe7", edgecolor=None, alpha = 0.4)

    return ax


def plot_ci_bootstrap(xs, ys, resid, nboot=500, ax=None):
    """Return an axes of confidence bands using a bootstrap approach.

    Notes
    -----
    The bootstrap approach iteratively resampling residuals.
    It plots `nboot` number of straight lines and outlines the shape of a band.
    The density of overlapping lines indicates improved confidence.

    Returns
    -------
    ax : axes
        - Cluster of lines
        - Upper and Lower bounds (high and low) (optional)  Note: sensitive to outliers

    References
    ----------
    .. [1] J. Stults. "Visualizing Confidence Intervals", Various Consequences.
       http://www.variousconsequences.com/2010/02/visualizing-confidence-intervals.html

    """
    if ax is None:
        ax = plt.gca()

    bootindex = scipy.random.randint

    for _ in range(nboot):
        resamp_resid = resid[bootindex(0, len(resid) - 1, len(resid))]
        # Make coeffs of for polys
        pc = scipy.polyfit(xs, ys + resamp_resid, 1)
        # Plot bootstrap cluster
        ax.plot(xs, scipy.polyval(pc, xs), "b-", linewidth=2, alpha=3.0 / float(nboot))

    return ax

def plot_scatter(x, y, title, xlabel, ylabel, limi, spinexy = True, font = 35, markersize = 2, alpha = 1, perc = 0.95, hist = 0, xtic = 0, showr2 = 0, showfit = 1):
    # plt.rcParams['font.family'] = 'Lucida Grande'
    sns.set(style="ticks", font_scale=2)
    # sns.set_style({'font.family':'serif', 'font.serif':'Helvetica', 'font.weight':'normal'})
    x = np.array(x)
    y = np.array(y)

    def func(x, a, b):
        return a * x + b

    popt, pcov = curve_fit(func, x, y)
    # r2 = r2_score(np.array(x), np.array(y))
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    skr2 = r2_score(x, y)


    p, cov = np.polyfit(x, y, 1, cov=True)                     # parameters and covariance from of the fit of 1-D polynom.
    y_model = equation(p, x)
    n = y.size                                           # number of observations
    m = p.size                                                 # number of parameters
    dof = n - m                                                # degrees of freedom
    t = stats.t.ppf(perc, n - m)
    print(p)
    fig, ax = plt.subplots(figsize=(12, 12))
    # ax = fig.add_subplot(111)
    if spinexy:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    else:
        for key, spine in ax.spines.items():
            spine.set_visible(False)
    # Data
    # ax.plot(
    #     x, y, "o", color="#b9cfe7", markersize=8,
    #     markeredgewidth=1, markeredgecolor="b", markerfacecolor="None"
    # )
    from matplotlib import colors
    if hist:
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        counts, xedges, yedges, im = ax.hist2d(x, y, bins = 50, cmap = 'Blues', density =  1, norm=colors.LogNorm(), vmin = 0.00001)
        ax2 = plt.gca()
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="2.5%", pad=0.01)
        # cbar_ax = fig.add_axes([0.9, 0.96, 0.01, 0.8])
        # fig.colorbar(thplot, cax=cbar_ax, orientation="horizontal")
        cbar=fig.colorbar(im, cax = cax)
        # cbar=fig.colorbar(im, aspect = 50)
        tick_font_size = font
        cbar.outline.set_linewidth(0.15)
        cbar.ax.tick_params(labelsize=tick_font_size)
        # plt.axis('scaled')
    else:
        ax.scatter(x, y, color = 'teal', s=markersize, alpha = 0.5)
    xx = [-30, limi]

    x2 = np.linspace(0, limi, 100)
    y2 = equation(p, x2)

    # Estimates of Error in Data/Model
    resid = y - y_model
    chi2 = np.sum((resid / y_model)**2)                        # chi-squared; estimates error in data
    chi2_red = chi2 / dof                                      # reduced chi-squared; measures goodness of fit
    s_err = np.sqrt(np.sum(resid**2) / dof)
    if showfit:
        plot_ci_manual(t, s_err, n, x, x2, y2, ax=ax)

    # # Prediction Interval
    # pi = t * s_err * np.sqrt(1 + 1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
    # ax.fill_between(x2, y2 + pi, y2 - pi, color="None", linestyle="--")
    # ax.plot(x2, y2 - pi, "--", color="0.5", label="95% Prediction Limits")
    # ax.plot(x2, y2 + pi, "--", color="0.5")
    plt.xlim(-30, limi)
    plt.ylim(-30, limi)

    plt.locator_params(axis='y', nbins=3)
    plt.locator_params(axis='x', nbins=3)
    if xtic:
        plt.xticks(fontsize=font)
    else:
        plt.xticks([])
    plt.yticks(fontsize=font)
    # plt.title(title, fontsize = 16)
    if showr2:
        if showfit:
            ax.plot(np.array(xx), func(np.array(xx), *popt), 'teal', label='f(x) = %5.2f x + %5.2f\n$Pearson  r$ = %5.2f' % (popt[0], popt[1], r_value))
        else:
            ax.text(10, 460, '$R^2$ = %5.2f'%(skr2), size = font)
    else:
        ax.plot(np.array(xx), func(np.array(xx), *popt), 'teal', label='f(x) = %5.2f x + %5.2f' % (popt[0], popt[1]))

    ax.plot(xx, xx, '--', color = 'gray', alpha = alpha)
    # L = ax.legend(fontsize = 22,)
    # plt.setp(L.texts, family='DejaVu Sans')
    if showfit:
        ax.legend(loc = 'upper left', fontsize = font-1, handlelength=0.5)
    if xtic:
        ax.yaxis.get_major_ticks()[0].label1.set_visible(False)

    plt.xlabel(xlabel,fontsize=font)
    plt.ylabel(ylabel,fontsize=font)

    return




def mae_group(intervals, preds, gtts):
    maes = []
    for i in range(len(intervals)-1):
        curgt = gtts[i]
        curpred = preds[i]
        assert len(curgt) == len(curpred)
        print(len(curgt))
        try:
            maei = mean_absolute_error(curgt, curpred)
            maes.append(maei)
        except:  # no higher than this range
            maes.append(0)

    return maes

def plot_box(ppd, gtt, gtts, preds, font = 35, title = None, spinexy = 0, lim =320, ylim = 220,
                label_nas = ['0-50', '50-100', '100-150', '150-200', '200-250',' 250-300', '>300']):
    # rc('font', weight='bold')
    # label_nas = ['1', '2', '3', '4', '5', '6', '7']
    labels = [0, 50, 100, 150, 200, 250, 300]
    # labels_left = [i-10 for i in labels]
    # labels = [0, 1, 2, 3, 4, 5, 6]
    sns.set(style="ticks", font_scale=2)
    # sns.set_style({'font.family':'serif', 'font.serif':'Helvetica', 'font.weight':'normal'})
    fig, ax = plt.subplots(figsize = (12, 12))
    ax.invert_yaxis()
    ax.hist(gtt, bins=50, color='tan', alpha=0.4, density = 1, label = 'Reference')
    ax.locator_params(axis='y', nbins=4)
    quartiles = [2.5, 97.5]
    for q in np.percentile(gtt, quartiles):
        plt.axvline(q, ls = '--', lw = 3, color='firebrick', alpha = 0.6)
    ax.tick_params(axis='y')
    ax.hist(ppd, bins=50, color='#89979a', alpha=0.3, density = 1, label = 'Prediction')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # ax.set_ylabel('MSD')
    plt.xlim(-20, lim)

    # plt.xticks(rotation=45)

    # plt.locator_params(axis='x', nbins=3)
    plt.xticks(fontsize=font)
    plt.yticks(fontsize=font)
    # ax.set_title('Mean Squared Deviation (MSE) by height ranges')
    # ax.legend(bbox_to_anchor=(0.9,0.99), prop={'size': 12})
    # ax2.legend(bbox_to_anchor=(0.2,0.99), prop={'size': 12})

    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis

    # ipdb.set_trace()
    # remove gt < 1 samples, they cause problems for error calculation
    # preds[0] = preds[0][gtts[0]>=1]
    # gtts[0] = gtts[0][gtts[0]>=1]
    maes = []
    for i in range(7):

        maes.append(abs(gtts[i] - preds[i]))
        # maes.append(abs((gtts[i] - preds[i])/(gtts[i]))) # use relative mae
    # ipdb.set_trace()
    print(len(maes))
    # boxprops=dict( color='teal')
    boxprops=dict( color='chocolate', linewidth=5)
    line_props = dict(linestyle='--', linewidth = 3, color="grey", alpha=1)
    capprops = dict(linewidth = 5, color="grey", alpha = 1)
    medianprops = dict(linewidth=5, color = 'purple')
    # meanlineprops = dict(linestyle='solid', linewidth=3.5, color='darkorange')
    meanlineprops = dict(linestyle='solid', linewidth=5, color='green')
    bp = ax2.boxplot( maes, widths = 20, showfliers=False, showmeans = 1,  meanprops=meanlineprops, meanline=True, patch_artist=True,
                        whiskerprops=line_props, boxprops = boxprops, capprops = capprops, medianprops=medianprops, positions = labels)
    # ax_box_x.set_title('Max height per tree - Errors at 10m height intervals', fontsize = 16)
    # fill with colors

    for patch in bp['boxes']:
        # patch.set_facecolor(color)
        # patch.set(facecolor = 'cadetblue', alpha = 0.6 )
        patch.set(facecolor = 'chocolate', alpha = 0.6 )

    # # add anther barplot
    # ax22 = ax2.twinx()
    # bp2 = ax22.boxplot( maes, widths = 10, showfliers=False, showmeans = 1,  meanprops=meanlineprops, meanline=True, patch_artist=True,
    #                     whiskerprops=line_props, boxprops = boxprops, capprops = capprops, medianprops=medianprops, positions = labels)
    #
    # for patch in bp2['boxes']:
    #     # patch.set_facecolor(color)
    #     # patch.set(facecolor = 'cadetblue', alpha = 0.6 )
    #     patch.set(facecolor = 'gray', alpha = 0.6 )

    ax2.set_xticks(labels)
    ax2.set_xticklabels(label_nas)
    ax2.set_ylim(-5, ylim)
    ax.yaxis.set_ticks_position('right')
    ax2.yaxis.set_ticks_position('left')
    # ax.set_yticklabels(fontsize=15)
    ax.yaxis.set_tick_params(labelsize=font)
    ax2.yaxis.set_tick_params(labelsize=font)
    ax2.locator_params(axis='y', nbins=4)
    if spinexy:
        # ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
    else:
        for key, spine in ax.spines.items():
            spine.set_visible(False)
        for key, spine in ax2.spines.items():
            spine.set_visible(False)

    # # subgroup total bias
    # rr = []
    # for i in range(len(gtts)):
    #     curgt = gtts[i]
    #     curpd = preds[i]
    #     ate = (curgt-curpd).sum()
    #     rte = ate/curgt.sum()
    #     rr.append(rte)
    #
    # ax3 = ax2.twinx()
    # ax3.plot(labels, rr, color = 'red')
    # ax3.spines['left'].set_position(('outward', 60))




    plt.show()

    return


def hex_to_rgb_color_list(colors):
    """
    Take color or list of hex code colors and convert them
    to RGB colors in the range [0,1].
    Parameters:
        - colors: Color or list of color strings of the format
                  '#FFF' or '#FFFFFF'
    Returns:
        The color or list of colors in RGB representation.
    """
    if isinstance(colors, str):
        colors = [colors]

    for i, color in enumerate(
        [color.replace('#', '') for color in colors]
    ):
        hex_length = len(color)

        if hex_length not in [3, 6]:
            raise ValueError(
                'Colors must be of the form #FFFFFF or #FFF'
            )

        regex = '.' * (hex_length // 3)
        colors[i] = [
            int(val * (6 // hex_length), 16) / 255
            for val in re.findall(regex, color)
        ]

    return colors[0] if len(colors) == 1 else colors


def blended_cmap(rgb_color_list):
    """
    Created a colormap blending from one color to the other.
    Parameters:
        - rgb_color_list: A list of colors represented as [R, G, B]
          values in the range [0, 1], like [[0, 0, 0], [1, 1, 1]],
          for black and white, respectively.
    Returns:
        A matplotlib `ListedColormap` object
    """
    if not isinstance(rgb_color_list, list):
        raise ValueError('Colors must be passed as a list.')
    elif len(rgb_color_list) < 2:
        raise ValueError('Must specify at least 2 colors.')
    elif (
        not isinstance(rgb_color_list[0], list)
        or not isinstance(rgb_color_list[1], list)
    ) or (
        len(rgb_color_list[0]) != 3 or len(rgb_color_list[1]) != 3
    ):
        raise ValueError(
            'Each color should be represented as a list of size 3.'
        )

    N, entries = 256, 4 # red, green, blue, alpha
    rgbas = np.ones((N, entries))

    segment_count = len(rgb_color_list) - 1
    segment_size = N // segment_count
    remainder = N % segment_count # need to add this back later

    for i in range(entries - 1): # we don't alter alphas
        updates = []
        for seg in range(1, segment_count + 1):
            # determine how much needs to be added back to account for remainders
            offset = 0 if not remainder or seg > 1 else remainder

            updates.append(np.linspace(
                start=rgb_color_list[seg - 1][i],
                stop=rgb_color_list[seg][i],
                num=segment_size + offset
            ))

        rgbas[:,i] = np.concatenate(updates)

    return ListedColormap(rgbas)


def draw_cmap(cmap, values=np.array([[0, 1]]), **kwargs):
    """
    Draw a colorbar for visualizing a colormap.
    Parameters:
        - cmap: A matplotlib colormap
        - values: The values to use for the colormap, defaults to [0, 1]
        - kwargs: Keyword arguments to pass to `plt.colorbar()`
    Returns:
        A matplotlib `Colorbar` object, which you can save with:
        `plt.savefig(<file_name>, bbox_inches='tight')`
    """
    img = plt.imshow(values, cmap=cmap)
    cbar = plt.colorbar(**kwargs)
    img.axes.remove()
    return cbar



def load_data(ppd, gtt, sample = 0):
    ppd = np.nan_to_num(ppd)
    gtt = np.nan_to_num(gtt)
    gtm = int(np.ceil(gtt.max()))

    inds = []
    intervals = [0, 50, 100, 150, 200, 250, 300, gtm]
    for i in range(len(intervals)-1):
        indi = [idx for idx,val in enumerate(gtt) if intervals[i] <= val < intervals[i+1]]
        inds.append(indi)


    preds = []
    gtts = []
    for i in range(len(intervals)-1):
        predi = ppd[inds[i]]
        preds.append(predi)
        gtti = gtt[inds[i]]
        gtts.append(gtti)

    return ppd, gtt, preds, gtts, intervals, gtm


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        # self.avg = self.sum / self.count
        self.avg = div0(self.sum,self.count)

def get_metrics(metrics, stage):
    """ Returns a dictionary of all metrics and losses being tracked
    """
    met = {}
    met[f"{stage}_rmse"] = metrics['rmse']
    met[f"{stage}_loss"] = metrics['loss']
    met[f"{stage}_r2"] = metrics['r2']

    # metrics[f"{self._stage}_total_rmse"] = self._rmse[-1].value()
    # metrics[f"{self._stage}_total_mae"] = self._mae[-1].value()
    return met

def get_metrics_classification(metrics, stage):
    """ Returns a dictionary of all metrics and losses being tracked
    """
    met = {}
    met[f"{stage}_f1"] = metrics['f1']
    # met[f"{stage}_acc"] = metrics['acc']
    met[f"{stage}_sens"] = metrics['sens']

    # metrics[f"{self._stage}_total_rmse"] = self._rmse[-1].value()
    # metrics[f"{self._stage}_total_mae"] = self._mae[-1].value()
    return met


def div0(x,y):
    try:
        return x/y
    except ZeroDivisionError:
        return 0


def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

# residual plots

def plot_resi(preds, lb):

    residuals = lb-preds
    # print(residuals.shape)

    # print(preds.shape)
    plt.figure()
    plt.axhline(y=0, color='r', linestyle='-')
    plt.scatter(preds, residuals)
    plt.xlabel('Predicted biomass (Mg/ha)', fontsize = 15)
    plt.ylabel('Target - Predicted biomass (Mg/ha)', fontsize = 15)

    plt.title('Residual plot for predictions\n sample no.= %d'%len(preds))

    return
