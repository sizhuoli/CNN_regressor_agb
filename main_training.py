#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 14:13:37 2022

@author: sizhuo
"""

import argparse
import os
from core.trainer import Trainer
import glob
import ipdb
# os.environ["CUDA_VISIBLE_DEVICES"]="1" # first gpu # not working well
import torch
import yaml
print('gpu check available', torch.cuda.is_available())
print('total device count', torch.cuda.device_count())
# set current device
torch.cuda.set_device(0) # 1 or 0
print('current device', torch.cuda.current_device())

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from omegaconf import OmegaConf



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--head', type=str, default='B', help='head type: A (dense) or B (map)')

    args = parser.parse_args()
    print('head design', args.head)
    if args.head == 'A':
        cfg = '/home/sizhuo/Desktop/code_repository/CNN_regressor_agb/conf/config_training_regression_head_A-dense-design.yaml'
    elif args.head == 'B':
        cfg = '/home/sizhuo/Desktop/code_repository/CNN_regressor_agb/conf/config_training_regression-head-B-map-design.yaml'
    else:
        print('wrong head type')
        raise ValueError

    conf = OmegaConf.load(cfg)
    Trainer(conf, cfg, showImOnly =0, launch_wandb = 1)


