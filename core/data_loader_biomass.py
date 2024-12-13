#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 12:23:49 2022

@author: sizhuo
"""

import os
import random
from random import shuffle
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torch.nn import FeatureAlphaDropout
from PIL import Image
import rasterio
import glob
import pandas as pd
from torch.utils.data.sampler import WeightedRandomSampler
# from skimage.transform import resize
from scipy.ndimage import zoom
import cv2
import matplotlib.pyplot as plt
from random import sample
from scipy import ndimage
import ipdb
import geopandas as gps
from skimage.transform import resize

# load one or multiple input image sources, concatenate them (np array), convert to tensor image using ToTensor() and then feed
# into torch vision transformer (takes tensor image of shape (B, C, H, W))
def NDVI(r, n):
    """
    NDVI function.
    Inputs:
        * r - red array (np.array)
        * n - nir array (np.array)
    Output:
        * ndvi - ndvi array (np.array)
    """

    np.seterr(divide='ignore', invalid='ignore') # Ignore the divided by zero or Nan appears
    n = np.float32(n)
    r = np.float32(r)
    ndvi = (n-r)/(n+r) # The NDVI formula
    ndvi = np.float32(ndvi) # Convert datatype to float32 for memory saving.

    return (ndvi)


def EXGI(g, r, b):
    exgi = 2 * g -(r + b)
    exgi = np.float32(exgi)
    return exgi

def rgb2gray(rgb):

    r, g, b = rgb[0, :,:], rgb[1, :,:], rgb[2, :,:]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def warpAffine(src, M, dsize, from_bounding_box_only=False):
    """
    Applies cv2 warpAffine, marking transparency if bounding box only
    The last of the 4 channels is merely a marker. It does not specify opacity in the usual way.
    """
    return cv2.warpAffine(src, M, dsize)


def rotate_image(image, angle):
    """Rotate the image counterclockwise.
    Rotate the image such that the rotated image is enclosed inside the
    tightest rectangle. The area not occupied by the pixels of the original
    image is colored black.
    Parameters
    ----------
    image : numpy.ndarray
        numpy image
    angle : float
        angle by which the image is to be rotated. Positive angle is
        counterclockwise.
    Returns
    -------
    numpy.ndarray
        Rotated Image
    """
    # get dims, find center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    image = warpAffine(image, M, (nW, nH), False)

    # image = cv2.resize(image, (w,h))

    return image

class ImageFolder(data.Dataset):
    def __init__(self, root, config, mode='train', split_list = []):
        """Initializes image paths and preprocessing module."""
        if type(root)!=str:
            self.root1 = root[0] # when having more data folders, the original data
        else: # when only one data source is used
            self.root1 = root
        # root= path to the images already split into train, val
        self.mode = mode
        self.config = config
        self.label_root = self.root1 + '../' # label file in the parent folder of separate data folders

        radius = int(self.config.croplength/2)
        self.mask = create_circular_mask(radius*2, radius*2, radius=radius)

        # print('normalize input------')
        # print('imageNet norm: ', self.config.imageNetnorm)
        # print('meanstd norm: ', self.config.meanstdnorm)

        print('===========================================Loading dataset 1 (original data)============================================')
        if self.config.add_chmInput or self.config.add_activ_loss_descend or self.config.add_activ_14reso:
            # import ipdb
            # ipdb.set_trace()
            if type(root)!=str: # more than one data source
                self.chm_paths = eval(f'config.{self.mode}_chm_path')[0]
            else:
                self.chm_paths = eval(f'config.{self.mode}_chm_path')

            self.chm_resample = self.config.chm_resample
            if self.config.chm_pred:
                chm_files = glob.glob(f'{self.chm_paths}/*chm.tif')
                chm_files = [os.path.basename(f).split('.')[0][:-9] for f in chm_files]
            elif not self.config.chm_pred:
                chm_files = glob.glob(f'{self.chm_paths}/*.tif')
                chm_files = [os.path.basename(f).split('.')[0] for f in chm_files]


        if self.config.add_seg:
            raise NotImplementedError('not implemented yet')

        else:
            if config.dk:
                # dk dataset
                if self.config.conf_score:
                    if self.mode != 'test':
                        if not self.config.mergeTrainValid:
                            if self.mode == 'train':
                                self.GT_dd = pd.read_csv(self.label_root + mode + '_split_scores_yeardiff_numtreesSameYear.csv')
                            elif self.mode == 'valid':
                                self.GT_dd = pd.read_csv(self.label_root + mode + '_split_scores_yeardiff_numtreesSameYear2.csv')
                        else:
                            #merge train and valid sets
                            self.GT_dd = pd.read_csv(self.label_root + 'train_valid_split_scores_yeardiff_numtreesSameYear.csv')


                    else:
                        self.GT_dd = pd.read_csv(self.label_root + mode + '_split_scores_numtreesSameYear2.csv')


                else:
                    self.GT_dd = pd.read_csv(self.label_root + mode + '_split.csv')

            else: # not dk
                # self.GT_dd = gps.read_file(self.config.GT_df_path)
                raise NotImplementedError('not implemented yet')



        # ipdb.set_trace()
        dd = self.GT_dd[self.GT_dd[self.config.attri_year] >= self.config.year]
        dd = dd[dd[self.config.attri_year] <= self.config.year_end]


        # normal case
        extract_fields = []
        extract_fields.extend([self.config.attri_id, self.config.attri_label, 'forest_type'])
        if self.config.conf_score:
            extract_fields.append(self.config.weight_name)
        if self.config.add_input2:
            extract_fields.append(self.config.input2_name)
        if self.config.correction_score:# the manual scores
            extract_fields.append(self.config.score_name)
        if self.config.add_outputs:
            extract_fields.extend(self.config.add_output_names)
        if self.config.test_type_separate:
            extract_fields.append(self.config.test_type_col)
        # weight for testing: forest fraction
        if config.dk:
            extract_fields.append(self.config.test_weight_name)
        extract_fields.append(self.config.attri_year)
        extract_fields = list(set(extract_fields))
        print(extract_fields)
        self.GT_df = dd[extract_fields]

        if self.config.weight_name == 'forest_frac' and self.config.conf_score and self.config.weight_forest_yesno:
            # so no 0 weight samples based on forest frac
            # import ipdb
            # ipdb.set_trace()
            self.GT_df.loc[self.GT_df[self.config.weight_name]<=0.1, self.config.weight_name] = 0.1
            # self.GT_df.loc[self.GT_df[self.weight_name]==np.nan, self.weight_name] = 0.01
        self.GT_df = self.GT_df.dropna(subset=[self.config.attri_label])


        if config.dk:
            self.GT_df = self.GT_df[self.GT_df[self.config.weight_name]!=0]
        selected_fns = self.GT_df[self.config.attri_id].to_list()
        self.image_paths0 =  glob.glob(f'{self.root1}/*.tif')
        if config.dk:
            self.image_paths = [f for f in self.image_paths0 if os.path.basename(f).split('.')[0] in selected_fns]
        else:
            self.image_paths = [f for f in self.image_paths0 if int(os.path.basename(f).split('.')[0]) in selected_fns]

        if self.config.test_type_separate:
            # ipdb.set_trace()
            test_list = self.GT_df.loc[self.GT_df[self.config.test_type_col] == self.config.test_type_name, self.config.attri_id].to_list()
            # self.image_paths =  glob.glob(f'{root}/*.tif')
            self.image_paths = [f for f in self.image_paths if os.path.basename(f).split('.')[0] in test_list]

        # remove samples in the testing set and also train set
        self.image_paths = [f for f in self.image_paths if f not in split_list]

        self.RotationDegree = [0,90,180,270]
        if self.mode != 'test':
            self.augmentation_prob = self.config.augmentation_prob

        elif self.mode == 'test':
            self.augmentation_prob = 0

        print("image count in {} path :{}; with years between {} and {}".format(self.mode,len(self.image_paths), self.config.year, self.config.year_end))

        # import ipdb
        # ipdb.set_trace()
        # TODO: adding sample weights
        if self.config.add_chmInput or self.config.add_activ_loss_descend or self.config.add_activ_14reso:
            self.image_paths = [f for f in self.image_paths if os.path.basename(f).split('.')[0] in chm_files]
            print("Aftering double checking chm files, image count in {} path :{}; with years between {} and {}".format(self.mode,len(self.image_paths), self.config.year, self.config.year_end))

        if type(root)!=str: # if train path has two paths!
            self.root2 = root[1] # new data
            self.GT_paths2 = self.root2 + '../'
            print('===========================================Loading dataset 2 (extended data)============================================')
            if self.config.add_chmInput or self.config.add_activ_loss_descend or self.config.add_activ_14reso:
                # import ipdb
                # ipdb.set_trace()
                self.chm_paths2 = eval(f'config.{self.mode}_chm_path')[1]
                if self.config.chm_pred:
                    chm_files2 = glob.glob(f'{self.chm_paths2}/*chm.tif')
                    chm_files2 = [os.path.basename(f).split('.')[0][:-9] for f in chm_files2]
                elif not self.config.chm_pred:
                    chm_files2 = glob.glob(f'{self.chm_paths2}/*.tif')
                    chm_files2 = [os.path.basename(f).split('.')[0] for f in chm_files2]

            if self.mode != 'test':
                self.GT_dd = pd.read_csv(self.GT_paths2 + '*.csv')
            else:
                self.GT_dd = pd.read_csv(self.GT_paths2 + '*.csv')
            dd = self.GT_dd[self.GT_dd[self.config.attri_year] >= 2019]
            extract_fields = []
            extract_fields.extend([self.config.attri_id, self.config.attri_label])
            if self.mode != 'test' and self.config.conf_score:# the manual scores
                extract_fields.append(self.config.weight_name)
                extract_fields.append('useCase')
            extract_fields.append(self.config.attri_year)

            extract_fields = list(set(extract_fields))
            print(extract_fields)
            self.GT_df2 = dd[extract_fields]
            if self.mode == 'train':
                self.GT_df2 = self.GT_df2[(self.GT_df2[self.config.weight_name]!=0) & (self.GT_df2['useCase']=='train')]
            elif self.mode == 'valid':
                self.GT_df2 = self.GT_df2[(self.GT_df2[self.config.weight_name]!=0) & (self.GT_df2['useCase']=='valid')]
            selected_fns = self.GT_df2[self.config.attri_id].to_list()
            self.image_paths0 = glob.glob(f'{self.root2}/*.tif')

            self.image_paths2 = [f for f in self.image_paths0 if os.path.basename(f).split('.')[0] in selected_fns]

            print("Loaded extended data: image count in {} path :{}; with years after 2019".format(self.mode,len(self.image_paths2)))


            if self.config.add_chmInput or self.config.add_activ_loss_descend or self.config.add_activ_14reso:
                self.image_paths2 = [f for f in self.image_paths2 if os.path.basename(f).split('.')[0] in chm_files2]
                print("Aftering double checking chm files, image count in {} path :{}; with years after 2019".format(self.mode,len(self.image_paths2)))

            self.image_paths.extend(self.image_paths2)
        print('=============================================================================')
        if type(root)!=str:
            print('===========0-0=============Loaded %d datasets=========0-0===;-)==========='%(len(root)))
        else:
            print('===========0-0=============Loaded 1 dataset=========0-0===;-)===========')

        print('Dataset: ', self.mode)
        print('Total count of images: ', len(self.image_paths))
        # print(self.image_paths)


        if self.config.subsample_data:
            if self.mode != 'test':
                # randomly* subsample data used for Training
                self.image_paths = random.choices(self.image_paths, k=int(self.config.subsample_data_rat*len(self.image_paths)))
                print("Aftering subsampling, image count in {} path :{}; with years between {} and {}".format(self.mode,len(self.image_paths), self.config.year, self.config.year_end))
            else:
                print('No data subsamping! image count in {} path: {}'.format(self.mode,len(self.image_paths)))



    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        filename = os.path.basename(image_path).split('.')[0] # file name without .tif


        if self.config.use_color:
            image = rasterio.open(image_path).read() # channel first
            if self.config.planet_data:
                # band order is BGR, NIR
                # ipdb.set_trace()
                if self.config.planet_gb_norm: # for resize 220 stats
                    means = np.array([362.03, 526.72, 396.01, 3252.88]) # band order: BGRNIR
                    stds = np.array([112.55, 177.96, 213.57, 623.66])
                    image = np.transpose(image, (1, 2, 0)) # channel last
                    image = (image - means)/stds
                    image = np.transpose(image, (2, 0, 1)) # channel last
                if self.config.nir_rep_red:
                    # print('replacing red with nir')
                    image = image[[3, 1, 0, 2], :, :] # NIRGBR
                    # if not self.config.planet_gb_norm:
                        # image[[0], :, :] = image[[0], :, :]/10 # rescale nir again
                else:
                    image = image[[2, 1, 0, 3], :, :] # RGBNIR

                # if not self.config.planet_gb_norm:
                #     # if not gb norm, need to rescale planet data
                #     image = image/10 # rescale planet data

                image = image.astype('float32')

                image = np.transpose(image, (1, 2, 0)) # channel last
                if self.config.random_patch_crop:
                    crop_image_size = random.randint(200, 400)
                    # print(crop_image_size)
                    image = cv2.resize(image, (crop_image_size, crop_image_size))
                else:
                    image = cv2.resize(image, (self.config.image_size, self.config.image_size))
                # crop
                image = center_crop(image, (self.config.croplength, self.config.croplength), channel_last=True)

                image = np.transpose(image, (2, 0, 1))

            # ipdb.set_trace()
            if self.config.add_ndvi:
                assert self.config.planet == 0
                ndvi = NDVI(image[0, :, :], image[-1, :, :])


            # if self.config.nir_rep_red:
            #     # replace nir with red
            #     image = image[[3, 1, 2, 0], :, :] # NIRGB

            if self.config.onlyRGB:
                image = image[:3, :, :]
                if self.config.rgb2gray:

                    if self.config.add_exgi:
                        # only when gary, exgi and chm bands_
                        exgi = EXGI(image[1, :, :], image[0, :, :], image[2, :, :])


                    image = rgb2gray(image)
                    if self.config.expandGray:
                        image = np.array([image]*2).astype(np.float32)
                    else:
                        image = np.array([image]*1).astype(np.float32)

                        if self.config.add_exgi:
                            exgi = np.array([exgi]*1).astype(np.float32)
                            image = np.concatenate((image, exgi), axis=0)
                        if self.config.add_ndvi:
                            ndvi = np.array([ndvi]*1).astype(np.float32)
                            image = np.concatenate((image, ndvi), axis=0)



        if self.config.add_chmInput:

            # randomly dropout chm layer
            rr = random.random()
            if rr > self.config.dropoutchm:
                chm = np.zeros((1, image.shape[1], image.shape[2])).astype(np.float32)


            else:
                try:
                    if self.root1 in image_path: # dataset 1
                        chm_path = image_path.replace(self.root1, self.chm_paths)
                    else: # dataset 2
                        chm_path = image_path.replace(self.root2, self.chm_paths2)
                    chm = rasterio.open(chm_path).read().astype(np.float32) # channel first
                except:
                    if self.root1 in image_path: # dataset 1
                        chm_path = image_path.replace(self.root1, self.chm_paths).replace('.tif', '_pred_chm.tif')
                    else:
                        chm_path = image_path.replace(self.root2, self.chm_paths2).replace('.tif', '_pred_chm.tif')
                    chm = rasterio.open(chm_path).read() # channel first
                    chm = chm.astype(np.float32)
                # chm = np.transpose(chm, axes=(1,2,0)) # Channel at the end
                chm[chm<0]=0
                chm[chm>50]=50
                # if self.config.chm_pred:
                #     chm = chm/2

                if self.config.chm_buffercrop: # chm for 200 m buffer area
                    # buffered chm firstly crop to same size as small img, then to center crop together with img
                    chm = center_crop(chm, (self.config.image_size, self.config.image_size), channel_last = 0)
                if self.config.chm_resample:
                    # usample chm
                    # chm = resize(chm, (int(chm.shape[0]*2), int(chm.shape[1]*2), 1), preserve_range=True)
                    # chm = chm * (chm.shape[0] / float(chm.shape[0]*2)) * (chm.shape[1] / float(chm.shape[1]*2))
                    # print('if channel last')
                    chm = zoom(chm, (1, 2,2), order=1)
                    # print('shp2', chm.shsplit_listape)


            if self.config.use_color:
                # image = np.concatenate((image, chm), axis=-1)
                try:
                    image = np.concatenate((image, chm), axis=0)
                except:
                    ipdb.set_trace()

            else: # only chm layer or ..
                image = chm

        if self.config.add_activ_loss_descend or self.config.add_activ_14reso:
            try:
                if self.root1 in image_path: # dataset 1
                    chm_path = image_path.replace(self.root1, self.chm_paths)
                else: # dataset 2
                    chm_path = image_path.replace(self.root2, self.chm_paths2)
                chm = rasterio.open(chm_path).read().astype(np.float32) # channel first
            except:
                if self.root1 in image_path: # dataset 1
                    chm_path = image_path.replace(self.root1, self.chm_paths).replace('.tif', '_pred_chm.tif')
                else:
                    chm_path = image_path.replace(self.root2, self.chm_paths2).replace('.tif', '_pred_chm.tif')
                chm = rasterio.open(chm_path).read() # channel first
                chm = chm.astype(np.float32)
            # chm = np.transpose(chm, axes=(1,2,0)) # Channel at the end
            chm[chm<0]=0
            chm[chm>50]=50

            #
            # chm_activ = zoom(chm, (1, 0.1,0.1), order=1)
            chm_ac = cv2.resize(chm.squeeze(), (14, 14))
            chm_activ = np.expand_dims(chm_ac, axis=0)
            # print('shp2', chm.shsplit_listape)

        if self.config.add_seg:
            seg_path = image_path.replace(self.root, self.seg_paths).replace('.tif', '_pred_seg.tif')
            seg = rasterio.open(seg_path).read()
            image = np.concatenate((image, seg), axis=0)



        if self.root1 in image_path: # dataset 1 # ensure sp_weight = True
            # print('######### dataset 1')
            GT, sp_weight = get_label_wei(self.config, self.GT_df, filename, self.mode)
        else: # dataset 2
            # print('========= dataset 2')
            GT, sp_weight = get_label_wei(self.config, self.GT_df2, filename, self.mode)
        if self.config.add_seg and self.config.refine_seg: # add sample confidence score to loss computation
            notree = self.GT_df.loc[self.GT_df[self.config.attri_id]==filename, 'notree'].tolist()
            if len(notree) != 1:
                notree = sum(notree) / len(notree)
            else:
                notree = notree[0]

        if self.config.add_input2:
            input2 = self.GT_df.loc[self.GT_df[self.config.attri_id]==filename, self.config.input2_name].tolist()
            if len(input2) != 1:
                input2 = sum(input2) / len(input2)
            else:
                input2 = input2[0]

            # use year diff
            input2 = (2018-input2)/3

        if self.config.add_outputs:
            add_ops = []
            for op in range(len(self.config.add_output_names)):
                output = self.GT_df.loc[self.GT_df[self.config.attri_id]==filename, self.config.add_output_names[op]].tolist()
                if len(output) != 1:
                    output = sum(output) / len(output)
                else:
                    output = output[0]
                add_ops.append(output)


        # crop here, now channel first
        if not self.config.planet_data:
            image_crop = center_crop(image, (self.config.croplength, self.config.croplength))
        else:
            image_crop = image

        # for resolution test
        if self.config.downsample:
            image_crop = np.transpose(image_crop, (1, 2, 0)) # channel last
            ori_size = image_crop.shape[1]
            image_crop = cv2.resize(image_crop, (int(ori_size/self.config.downsample_ratio), int(ori_size/self.config.downsample_ratio)))
            image_crop = cv2.resize(image_crop, (ori_size, ori_size))
            image_crop = np.transpose(image_crop, (2, 0, 1)) # channel first
            # ipdb.set_trace()
            # downsample input images

        def cropSquare(config, image, rand_x, rand_y):
            # image (C, H, W); C = 4
            # rand_x, rand_y in (0, 30)
            # length = min(croplength - i, croplength - j)
            length = min(config.croplength - rand_x, config.croplength - rand_y)
            return image[:, rand_x: rand_x + length, rand_y:rand_y + length]

        p_transform = random.random()

        # first augmentation: random crop
        if (self.mode == 'train') and self.config.cropRandomSquare:
            rand_x = random.randint(0, int(self.config.croplength-self.config.cropInnerLength))
            rand_y = random.randint(0, int(self.config.croplength-self.config.cropInnerLength))
            image_crop = cropSquare(self.config, image_crop, rand_x, rand_y)

            # ipdb.set_trace()


        if self.config.circle_crop:
            new_length = image_crop.shape[-1]
            rad2 = int(new_length/2)
            mask2 = create_circular_mask(rad2*2, rad2*2, radius=rad2)
            # mask2 = torch.from_numpy(mask2)
            # ipdb.set_trace()
            try:
                image_crop[:, ~mask2] = 0
            except:
                # odd cropLength_
                mask22 = np.pad(mask2, (1, 0), 'constant', constant_values=(0))
                # ipdb.set_trace()
                image_crop[:, ~mask22] = 0

        # to fed into totensor, need channel last
        image_crop = np.transpose(image_crop, axes=(1,2,0)) # Channel at the end


        # crop to circle first and then rotate, to maintain information
        if (self.mode == 'train') and p_transform <= self.config.augmentation_prob:
            if not self.config.planet_data:
                rot_deg = random.randint(0,360)
            else:
                rot_deg = random.choice(self.RotationDegree)
            image_crop = ndimage.rotate(image_crop, rot_deg, reshape=False)
        if self.config.add_seg and self.config.refine_seg:
            seg = image_crop[:, :, -1] # seg at the final channel
            seg2 = np.zeros(seg.shape)
            cnts, _ = cv2.findContours(seg.copy().astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            segcount = 0
            for ct in cnts:
                area = cv2.contourArea(ct)
                if area >= 20:
                    # cv2.drawContours(seg2, [ct], -1, (255, 0, 0), -1)
                    # M = cv2.moments(ct)
                    # cx = int(M['m10']/M['m00'])
                    # cy = int(M['m01']/M['m00'])
                    # if self.segmask[cx, cy] == 1: # centroid is in the mask
                    cv2.fillPoly(seg2, pts = [ct], color =1)
                    segcount += 1

            image_crop[:, :, -1] = seg2


        # normalize with only valid values in the crop

        if self.config.meanstdnorm:
            if self.config.add_chmInput and not self.config.onlyRGB: # norm only color
                image_crop_color = image_crop[:, :, :4]
                masked_crop = np.ma.masked_equal(image_crop_color, 0)
                normed_crop = (masked_crop-masked_crop.mean(axis = (0, 1)))/masked_crop.std(axis = (0, 1))
                image_crop[:, :, :4] = normed_crop.data
                image_crop = image_crop.astype(np.float32)
            else: # norm all bands
                masked_crop = np.ma.masked_equal(image_crop, 0)
                normed_crop = (masked_crop-masked_crop.mean(axis = (0, 1)))/masked_crop.std(axis = (0, 1))
                image_crop = normed_crop.data.astype(np.float32)


        if self.config.equalize_img:
            image_crop = T.ToPILImage()(image_crop)
            image_crop = T.functional.equalize(image_crop)
        # ipdb.set_trace()
        # # add rotation manually
        # if (self.mode == 'train') and p_transform <= self.config.augmentation_prob:
        #     ang = random.randint(300,320)
        #     image_crop = rotate_image(image_crop, 40)

        Transform = []
        if self.config.add_chmInput:
            Transform.append(T.ToTensor())
            if (self.mode == 'train') and p_transform <= self.config.augmentation_prob:

                Transform.append(T.RandomHorizontalFlip())
                Transform.append(T.RandomVerticalFlip())
                Transform = T.Compose(Transform)
                image_crop = Transform(image_crop)
                Transform =[]
                # Transform.append(T.ToTensor())


        else:
            if (self.mode == 'train') and p_transform <= self.config.augmentation_prob:

                Transform.append(T.ToPILImage())
                Transform.append(T.RandomHorizontalFlip())
                Transform.append(T.RandomVerticalFlip())
                # Transform.append(T.RandomEqualize(p=0.2))
                Transform.append(T.RandomRotation(180))
                Transform.append(T.RandomAdjustSharpness(sharpness_factor=1.1))
                # Transform.append(T.RandomAutocontrast())
                Transform.append(T.ColorJitter(brightness=.1, contrast = .1, saturation = .05, hue=.05))

                Transform = T.Compose(Transform)
                image_crop = Transform(image_crop)
                Transform =[]

            Transform.append(T.ToTensor())

        # Transform.append(T.ToTensor())
        Transform.append(T.Resize((self.config.inputlength, self.config.inputlength)))
        if self.config.imageNetnorm:
            Transform.append(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        Transform = T.Compose(Transform)
        image_crop = Transform(image_crop)
        # ipdb.set_trace()
        if self.config.dropoutchannel and self.mode == 'train':
            # randomly dropout an entire channel
            drop = FeatureAlphaDropout(p=self.config.dropoutchannel_ratio)
            image_crop = drop(image_crop)


        GT = torch.tensor(GT)
        if self.config.conf_score:

            sp_weight = torch.tensor(sp_weight)
            if self.config.add_input2:
                input2 = torch.tensor(input2)
                return image_crop, input2, GT, sp_weight
            elif self.config.add_outputs:
                add_ops = [torch.tensor(ap) for ap in add_ops]
                return image_crop, GT, *add_ops, sp_weight

            # elif 'test' in self.mode and self.config.test_vis:
            elif self.config.test_vis:
                
                if self.config.add_activ_loss_descend or self.config.add_activ_14reso:
                    return image_crop, GT, chm_activ, sp_weight, filename
                else:
                    return image_crop, GT, sp_weight, filename

            else:
                if self.config.add_activ_loss_descend or self.config.add_activ_14reso:
                    return image_crop, GT, chm_activ, sp_weight
                else:
                    return image_crop, GT, sp_weight

        else:
            # return image, GT
            return image_crop, GT

    def __len__(self):
        """Returns the total number of files."""
        return len(self.image_paths)



def get_loader(image_path, config, mode, split_list, biasC = 0):
    """Builds and returns Dataloader."""
    dataset = ImageFolder(root = image_path, config = config, mode = mode, split_list = split_list)

    # import ipdb
    # ipdb.set_trace()
    if mode != 'test' and not biasC:
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=config.batch_size,
                                      shuffle=True,
                                      )
    else:
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=config.batch_size,
                                      shuffle=False,
                                      )
        # print(len(data_loader))
    return data_loader


def get_label_wei(config, df, filename, mode):

    if config.dk:
        GT = df.loc[df[config.attri_id]==filename, config.attri_label].tolist()
    else:
        GT = df.loc[df[config.attri_id]==int(filename), config.attri_label].tolist()

    if len(GT) != 1:
        # GT = max(GT)
        # change date: June 29
        GT = sum(GT) / len(GT)
    else:
        GT = GT[0]

    if config.conf_score: # add sample confidence score to loss computation
        if mode != 'test':

            sp_weight = df.loc[df[config.attri_id]==filename, config.weight_name].tolist()
            if len(sp_weight) != 1:
                if max(sp_weight) - min(sp_weight) <= 0.3:
                    sp_weight = sum(sp_weight) / len(sp_weight)
                else:
                    sp_weight = 0.1
            else:
                sp_weight = sp_weight[0]

        else: # for test phase, do not use special weights
            sp_weight = 1

        if mode == 'valid' and not config.conf_score_valid:
            sp_weight = 1
    if config.conf_score:
        return GT, sp_weight
    else:
        return GT

def center_crop(img, crop_size, channel_last = 0):
    # Note: image_data_format is 'channel_last'
    # assert img.shape[2] == 4
    if channel_last:
        height, width = img.shape[0], img.shape[1]
    else:
        height, width = img.shape[1], img.shape[2]
    dy, dx = crop_size

    sx = (height - dx) // 2
    sy = (width - dy) // 2
    if channel_last:
        return img[sy:(sy+dy), sx:(sx+dx), :]
    else:
        return img[:, sy:(sy+dy), sx:(sx+dx)]


# def visualize_data()


def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask
