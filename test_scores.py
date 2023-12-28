#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 12:45:21 2022

@author: sizhuo
"""


import argparse
import os
import sys
sys.path.append(os.path.abspath("/home/sizhuo/Desktop/code_repository/CNN_AGB/"))
from core.trainer import Trainer
from core.solver_new import *
import glob
import ipdb
import torch
import yaml
print('gpu check available', torch.cuda.is_available())
print('total device count', torch.cuda.device_count())
# set current device
# torch.cuda.set_device(0) # 1 or 0
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
print('current device', torch.cuda.current_device())

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


cfg = './conf/config_testscore_rgb.yaml'


from omegaconf import OmegaConf

conf = OmegaConf.load(cfg)

res = Trainer(conf, cfg, showImOnly =0, launch_wandb = 0)
