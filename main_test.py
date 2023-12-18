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

cfg = './conf/config_old_data.yaml'
# cfg = './conf/config_old_data_with_height.yaml'
# cfg = './conf/config_old_data_attentive_selection.yaml'
# cfg = './conf/config_old_data_predchm2.yaml'
from omegaconf import OmegaConf
from sklearn.metrics import mean_squared_error ,mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt  # plotting tools
import wandb
conf = OmegaConf.load(cfg)

print(OmegaConf.to_yaml(conf))

Trainer(conf, cfg, showImOnly =0, launch_wandb = 1)


# ggt, ppd_b = Trainer(conf, showImOnly = 0)


# =============================================================================
# 
# =============================================================================

# gird testing (changing yaml files)
#
# out_branch = [#'_dense',
#                 '_map'
#               ]
#
#
#
# model_base = [
#      'torchEfficientnetb0',
#       #   'torchEfficientnetb1',
#       #   'torchEfficientnetb2',
#       # 'regnetY800MF',
#     # 'mobilenetv3',
#
#     ]
#
#
# # model_base = ['regnetY800MF',
# #                 'torchEfficientnetb2',
# #                 ]
#
#
# search_list = [p + d for p in model_base for d in out_branch]
# # search_list = search_list[1:]
#
#
#
# sts = [2017]
# ends = [2019, 2020]
#
#
#
#
#
#
# def set_state(mm, cfg, addchm = 0):
#     with open(cfg) as f:
#         doc = yaml.safe_load(f)
#
#     doc['model_type'] = mm
#
#     if addchm:
#         doc['add_chmInput'] = 1
#         if not doc['chm_pred']:
#             # lidar height
#             doc['chm_resample'] = 1
#         else:
#             doc['chm_resample'] = 0
#         doc['rgb2gray'] = 1
#         if doc['add_ndvi'] or doc['add_exgi']:
#             doc['expandGray'] = 0
#         else:
#             doc['expandGray'] = 1
#
#     else:
#         doc['add_chm'] = 0
#         doc['chm_resample'] = 0
#         doc['rgb2gray'] = 0
#         doc['expandGray'] = 0
#
#     with open(cfg, 'w') as f:
#         yaml.safe_dump(doc, f, default_flow_style=0)
#     return
#
# def set_state_reso(mm, ratio, cfg, addchm = 0):
#     with open(cfg) as f:
#         doc = yaml.safe_load(f)
#
#     doc['model_type'] = mm
#
#     if addchm:
#         doc['add_chm'] = 1
#         if not doc['chm_pred']:
#             # lidar height
#             doc['chm_resample'] = 1
#         else:
#             doc['chm_resample'] = 0
#         doc['rgb2gray'] = 1
#         if doc['add_ndvi'] or doc['add_exgi']:
#             doc['expandGray'] = 0
#         else:
#             doc['expandGray'] = 1
#
#     else:
#         doc['add_chm'] = 0
#         doc['chm_resample'] = 0
#         doc['rgb2gray'] = 0
#         doc['expandGray'] = 0
#
#     doc['downsample_ratio'] = ratio
#     doc['model_suf'] = 'downsample_ratio_' + str(ratio)
#     with open(cfg, 'w') as f:
#         yaml.safe_dump(doc, f, default_flow_style=0)
#     return
#
# def set_state_subsample(mm, ratio, cfg, addchm = 1):
#     with open(cfg) as f:
#         doc = yaml.safe_load(f)
#
#     doc['model_type'] = mm
#
#     if addchm:
#         doc['add_chm'] = 1
#         if not doc['chm_pred']:
#             # lidar height
#             doc['chm_resample'] = 1
#         else:
#             doc['chm_resample'] = 0
#         doc['rgb2gray'] = 1
#         if doc['add_ndvi'] or doc['add_exgi']:
#             doc['expandGray'] = 0
#         else:
#             doc['expandGray'] = 1
#
#     else:
#         doc['add_chm'] = 0
#         doc['chm_resample'] = 0
#         doc['rgb2gray'] = 0
#         doc['expandGray'] = 0
#
#     doc['subsample_data_rat'] = ratio
#     doc['model_suf'] = 'evened_val_test_subsample' + str(int(ratio*100))
#     with open(cfg, 'w') as f:
#         yaml.safe_dump(doc, f, default_flow_style=0)
#     return
#
# def set_state_year(year1, year2, cfg, addchm = 0):
#     with open(cfg) as f:
#         doc = yaml.safe_load(f)
#
#     doc['year'] = year1
#     doc['year_end'] = year2
#
#     if addchm:
#         doc['add_chm'] = 1
#         doc['chm_resample'] = 1
#         doc['rgb2gray'] = 1
#         if doc['add_ndvi'] or doc['add_exgi']:
#             doc['expandGray'] = 0
#         else:
#             doc['expandGray'] = 1
#
#     else:
#         doc['add_chm'] = 0
#         doc['chm_resample'] = 0
#         doc['rgb2gray'] = 0
#         doc['expandGray'] = 0
#
#     with open(cfg, 'w') as f:
#         yaml.safe_dump(doc, f, default_flow_style=0)
#     return
#
#
# def check_run(mm, wanpath = '/home/sizhuo/Desktop/code_repository/CNN_AGB/wandb/'):
#     all_subdirs = [d for d in glob.glob(f'{wanpath}/*/')
#                     if os.path.isdir(d)]
#
#     latest_subdir = sorted(all_subdirs, key=os.path.getmtime)[-1]
#
#     if latest_subdir.split('/')[-2] == 'latest-run':
#         latest_subdir = sorted(all_subdirs, key=os.path.getmtime)[-2]
#
#     run_id = latest_subdir.split('/')[-2].split('-')[-1]
#
#     run_path = 'sizli/biomass/' + run_id
#
#     api = wandb.Api()
#     currun = api.run(run_path)
#
#     # ipdb.set_trace()
#
#     if currun.config['model_name'] == mm:
#         if currun.summary['_step'] < 50:
#             print('!!!!!!!!!!!!! Not fininshed')
#             return 'rerun', 0
#         else:
#             print('heere')
#
#             return 'continue', (currun.summary['val_loss'], currun.summary['val_r2'])
#
#     else:
#         print('training error!!!!!!!!!!!!1')
#         return 'rerun', 0
#
#         # print('except///////////////////')
#         # return 'rerun', 0
#
#
#
# # c = 0
#
# # max_r2 = 0
# # min_loss = 1000
#
# # lossm = ''
# # r2m = ''
#
# for c in range(len(search_list)):
#     for re in range(1):
#         # if re == 0:
#         #     # print('skipping no height case')
#         #     set_state(search_list[c], cfg)
#         # else:
#         set_state(search_list[c], cfg, addchm = 1)
#
#         conf = OmegaConf.load(cfg)
#
#         print(OmegaConf.to_yaml(conf))
#
#         Trainer(conf, cfg, showImOnly = 0)
#         # stut, scores = check_run(search_list[c])
#         # while stut == 'rerun':
#         #     Trainer(conf, showImOnly = 0)
#         #     stut, scores = check_run(search_list[c])
#
#         # c += 1
#         # loss, r2 = scores
#
#         # if min_loss > loss:
#         #     min_loss = loss
#         #     lossm = search_list[c]
#
#         # if max_r2 < r2:
#         #     max_r2 = r2
#         #     r2m = search_list[c]
#
#         print('================================================================')
#
#
# print('finished grid search')
# print('-----------------------')
#
#
# # downsampling image ratio
# rats = [2, 4]
# rats =[8, 16]
# rats = [1]
# for c in range(len(search_list)):
#     for raa in rats:
#         for re in range(1):
#             set_state_reso(search_list[c], raa, cfg)
#
#             conf = OmegaConf.load(cfg)
#
#             print(OmegaConf.to_yaml(conf))
#
#             Trainer(conf, cfg, showImOnly = 0)
#
#             print('================================================================')
#
#
# print('finished grid search')
# print('-----------------------')
#
# # subsampling data amount ratio
# rats = [0.9, 0.8, 0.7, 0.6, 0.5]
# for c in range(len(search_list)):
#     for raa in rats:
#
#         set_state_subsample(search_list[c], raa, cfg, addchm = 0)
#
#         conf = OmegaConf.load(cfg)
#
#         print(OmegaConf.to_yaml(conf))
#
#         Trainer(conf, cfg, showImOnly = 0)
#
#         print('================================================================')
#
#
# print('finished grid search')
# print('-----------------------')
#
#
# for year1 in sts:
#     for year2 in ends:
#         for re in range(1):
#             if re == 0:
#                 set_state_year(year1, year2, cfg, addchm = 0)
#             else:
#                 set_state_year(year1, year2, cfg, addchm = 1)
#
#             conf = OmegaConf.load(cfg)
#
#             print(OmegaConf.to_yaml(conf))
#
#             Trainer(conf, cfg, showImOnly = 0)
#             # stut, scores = check_run(search_list[c])
#             # while stut == 'rerun':
#             #     Trainer(conf, showImOnly = 0)
#             #     stut, scores = check_run(search_list[c])
#
#             # c += 1
#             # loss, r2 = scores
#
#             # if min_loss > loss:
#             #     min_loss = loss
#             #     lossm = search_list[c]
#
#             # if max_r2 < r2:
#             #     max_r2 = r2
#             #     r2m = search_list[c]
#
#             print('================================================================')
#
#
# print('finished grid search')
# print('-----------------------')