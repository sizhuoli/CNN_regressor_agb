
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

cfg = './conf/config_inference_rgb.yaml'
from omegaconf import OmegaConf

conf = OmegaConf.load(cfg)

print(OmegaConf.to_yaml(conf))
#
# # ===== for quick small area test =====
pred = Predictor(conf)
pred.predict_all()
