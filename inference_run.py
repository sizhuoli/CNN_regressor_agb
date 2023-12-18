
# for inference only

import argparse
import os
from core.predictor import Predictor
import torch
import yaml
print('gpu check available', torch.cuda.is_available())
print('total device count', torch.cuda.device_count())
# set current device
torch.cuda.set_device(0) # 1 or 0
print('current device', torch.cuda.current_device())
import ipdb
import json

# cfg = './conf/config_inference2.yaml'
cfg = './conf/config_inference_rgb.yaml'
from omegaconf import OmegaConf
from sklearn.metrics import mean_squared_error ,mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt  # plotting tools

conf = OmegaConf.load(cfg)

print(OmegaConf.to_yaml(conf))
#
# # ===== for quick small area test =====
pred = Predictor(conf)
pred.predict_all()
#
ipdb.set_trace()

import glob
# ffs = glob.glob("/mnt/ssdc/Denmark/summer2018raw/*/", recursive = True)
# ffs = glob.glob('/media/RS_storage/Aerial/Denmark/summer2020/Final_Tiffjpeg/*/', recursive = True)
ffs = glob.glob('/mnt/ssdc/Denmark/summer2018raw/*/', recursive = True)
ffs.sort()

folders = [d[-6:-1] for d in ffs] 

def set_state(idd, cfg):
    with open(cfg) as f:
        doc = yaml.safe_load(f)
    old = doc['input_image_dir'][-5:]
    doc['input_image_dir'] = doc['input_image_dir'].replace(old, idd)
    doc['output_dir'] = doc['output_dir'].replace(old, idd)
    print(doc['input_image_dir'])
    with open(cfg, 'w') as f:
        yaml.safe_dump(doc, f, default_flow_style=0)
    return


#
# folders = [#'82_19', '82_20', '82_21', '83_25', '83_26']
#     '83_27', '83_28', '83_30',
#         '83_31', '83_32']


# folders.extend([d[-6:-1] for d in ffs[:33]] )

ipdb.set_trace()
for fd_id in folders:
    set_state(fd_id, cfg)
    conf = OmegaConf.load(cfg)
    
    # print(OmegaConf.to_yaml(conf))
    pred = Predictor(conf)
    c, notwork, nochm = pred.predict_all()
    dicc = {'notworking files': notwork, 'nochm files': nochm}
    a_file = open(conf.output_dir + '/pred_summary.json', 'w+')
    json.dump(dicc, a_file)
    a_file.close()


