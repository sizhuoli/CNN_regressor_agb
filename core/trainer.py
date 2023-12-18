#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 17:27:20 2022

@author: sizhuo
"""

from core.solver_new import Solver
from core.data_loader_biomass import get_loader
# from core.data_loader_biomass_test2020 import get_loader

# from core.data_loader_classification import get_loader
from torch.backends import cudnn
import random
import matplotlib.pyplot as plt  # plotting tools
import numpy as np
import glob
from sklearn.model_selection import train_test_split
import json
import time
import pandas as pd
import os
import ipdb
from tifffile import imsave
def Trainer(config, cfg_path, showImOnly = 0, launch_wandb = 1):
    # cudnn.benchmark = True
    # if config.model_type not in ['U_Net','R2U_Net','AttU_Net','R2AttU_Net']:
    #     print('ERROR!! model_type should be selected in U_Net/R2U_Net/AttU_Net/R2AttU_Net')
    #     print('Your input for model_type was %s'%config.model_type)
    #     return

    # # Create directories if not exist
    # if not os.path.exists(config.model_path):
    #     os.makedirs(config.model_path)
    # if not os.path.exists(config.result_path):
    #     os.makedirs(config.result_path)
    # config.result_path = os.path.join(config.result_path,config.model_type)
    # if not os.path.exists(config.result_path):
    #     os.makedirs(config.result_path)

    # lr = random.random()*0.0005 + 0.0000005
    # augmentation_prob= random.random()*0.7
    # epoch = random.choice([100,150,200,250])
    # decay_ratio = random.random()*0.8
    # decay_epoch = int(epoch*decay_ratio)

    # config.augmentation_prob = augmentation_prob
    # config.num_epochs = epoch
    # config.lr = lr
    # config.num_epochs_decay = decay_epoch

    print(config)

    # if merged train and valid, need to split first
    if config.mergeTrainValid:
        if config.task == 'regression':
            if not config.updated_NFI:
                GT_dd = pd.read_csv(config.train_path + '../train_valid_split_scores_yeardiff_numtreesSameYear.csv')
                print('GTdd', len(GT_dd))
                dd = GT_dd[GT_dd['year'] >= config.year]
                print('step1', len(dd))
                dd = dd[dd[config.weight_name]!=0]
                print('step2', len(dd))
                selected_fns = dd['psussu'].to_list()
                all_files = glob.glob(f'{config.train_path}/*.tif')
                print('all', len(all_files))
                image_paths = [f for f in all_files if os.path.basename(f).split('.')[0] in selected_fns]
                print('sel', len(image_paths))
                train_list, valid_list = train_test_split(image_paths, test_size=1-config.split_ratio)
                print('train', len(train_list), len(valid_list))
                timestr = time.strftime("%Y%m%d-%H%M")
                json_file = config.train_path + '../trainValid_split/' + 'split_' + timestr + '_split_' + str(int(config.split_ratio*10)) + '.json'
                data = {}
                data['training'] = train_list
                data['validation'] = valid_list
                data['num_train'] = len(train_list)
                data['num_valid'] = len(valid_list)
                data['train_ratio'] = config.split_ratio
                with open(json_file, 'w') as file:
                    json.dump(data, file)
            else:
                # new dataset
                GT_dd = pd.read_csv(config.train_path + '../NFIdata2021_LIDAR_scores_traintest.csv')
                print('GTdd', len(GT_dd))
                selected_fns = GT_dd.loc[(GT_dd['year']>=config.year)&(GT_dd['useCase']=='train'), 'psussu'].to_list()
                print('sel len', len(set(selected_fns)))
                all_files = glob.glob(f'{config.train_path}/*.tif')
                print('all', len(all_files))
                image_paths = [f for f in all_files if os.path.basename(f).split('.')[0] in selected_fns]
                print('sel', len(image_paths))
                train_list, valid_list = train_test_split(image_paths, test_size=1-config.split_ratio)
                print('train', len(train_list), len(valid_list))
                timestr = time.strftime("%Y%m%d-%H%M")
                json_file = config.train_path + '../trainValid_split/' + 'split_' + timestr + '_split_' + str(int(config.split_ratio*10)) + '.json'
                data = {}
                data['training'] = train_list
                data['validation'] = valid_list
                data['num_train'] = len(train_list)
                data['num_valid'] = len(valid_list)
                data['train_ratio'] = config.split_ratio
                with open(json_file, 'w') as file:
                    json.dump(data, file)

        elif config.task == 'classification':
            GT_dd = pd.read_csv(config.classification_label_fn)
            print('GTdd', len(GT_dd))
            selected_fns = GT_dd.loc[GT_dd['useCase']=='train', 'path'].to_list()
            print('sel len', len(set(selected_fns)))
            all_files = glob.glob(f'{config.train_path}/**/*.tif', recursive=True)
            print('all', len(all_files))
            if config.task == 'regression':
                image_paths = [f for f in all_files if os.path.basename(f).split('.')[0] in selected_fns]
            else:
                image_paths = [f for f in all_files if f in selected_fns]
            print('sel', len(image_paths))
            train_list, valid_list = train_test_split(image_paths, test_size=1-config.split_ratio)
            print('train', len(train_list), len(valid_list))
            timestr = time.strftime("%Y%m%d-%H%M")
            json_base = config.train_path + 'trainValid_split/'
            if not os.path.exists(json_base):
                os.mkdir(json_base)
            json_file = json_base + 'split_' + timestr + '_split_' + str(int(config.split_ratio*10)) + '.json'
            data = {}
            data['training'] = train_list
            data['validation'] = valid_list
            data['num_train'] = len(train_list)
            data['num_valid'] = len(valid_list)
            data['train_ratio'] = config.split_ratio

            with open(json_file, 'w') as file:
                json.dump(data, file)

        else:
            print('task invalid')


    else:
        train_list = valid_list = []



    train_loader = get_loader(image_path=config.train_path, config = config,
                            split_list = train_list,
                            mode='train')
    train_loader_fixed = get_loader(image_path=config.train_path, config = config,
                            split_list = train_list,
                            mode='train', biasC = 1)
    # # for bias correction
    # elif config.mode == "biasCorrect":
    #     train_loader = get_loader(image_path=config.train_path,
    #                               use_color = config.use_color,
    #                               onlyRGB = config.onlyRGB,
    #                               add_chm = config.add_chm,
    #                               chm_pred = config.chm_pred,
    #                               dropoutchm = config.dropoutchm,
    #                               chm_path = config.train_chm_path,
    #                               chm_resample = config.chm_resample,
    #                               add_seg = config.add_seg, seg_path = config.train_seg_path,
    #                               notree = config.notree,
    #                               conf_score = config.conf_score,
    #                               weight_name = config.weight_name,
    #                               imageNetnorm = config.imageNetnorm,
    #                              meanstdnorm = config.meanstdnorm,
    #                               yearaft = config.year,
    #                               croplength = config.croplength,
    #                               inputlength = config.inputlength,
    #                               radius = int(config.croplength/2),
    #                             image_size=config.image_size,
    #                             batch_size=config.batch_size,
    #                             num_workers=config.num_workers,
    #                             mode='train',
    #                             augmentation_prob=0., biasC = 1)

    valid_loader = get_loader(image_path=config.valid_path,
                              config = config,
                              split_list = valid_list,
                            mode='valid')

    test_loader = get_loader(image_path=config.test_path,
                            config = config,
                            split_list = train_list + valid_list,
                            mode='test')

    # import ipdb
    # ipdb.set_trace()
    if config.mode == 'train':
        print('Checking training data')

        images = data_vis(train_loader, config.notree, config.conf_score, config.use_color, config.add_chmInput, config.add_seg, add_activ_loss_descend = max(config.add_activ_loss_descend, config.add_activ_14reso), twoinput = config.add_input2, outputs = config.add_outputs)

        print('Checking validation data')
        _ = data_vis(valid_loader, config.notree, config.conf_score, config.use_color, config.add_chmInput, config.add_seg, add_activ_loss_descend = max(config.add_activ_loss_descend, config.add_activ_14reso), twoinput = config.add_input2, outputs = config.add_outputs)

    else:

        images = data_vis(test_loader, config.notree, config.conf_score, config.use_color, config.add_chmInput, config.add_seg, add_activ_loss_descend = max(config.add_activ_loss_descend, config.add_activ_14reso), num = 1 , test_vis=config.test_vis)
        # images = data_vis(test_loader, config.notree, config.conf_score, config.use_color, config.add_chmInput, config.add_seg, num = 1, model = solver.nnet, devi = solver.device, test = 1)
        print('***')
    if config.add_input2:
        patch_size = [images[0].shape, images[1].shape]
    else:
        patch_size = images.shape
    if not showImOnly:
        # print('patch size:', patch_size)
        solver = Solver(config, train_loader, valid_loader, test_loader, patch_size, cfg_path, launch_wandb = launch_wandb)

        # Train and sample the images
        if config.mode == 'train':
            solver.train()
        elif config.mode == 'train_transfer':
            solver.train_transfer()
        elif config.mode == 'AttentiveSelection':
            solver.AttentiveSelection()
        elif config.mode == 'biasCorrect':
            solver.bias_correction(train_loader_fixed, valid_loader)
        elif config.mode == 'bias_test':
            solver.bias_correction(train_loader_fixed, valid_loader)
            if config.test_vis:
                ppd, gtt, gtts, preds, mae, r2, mse, rte, filenames = solver.test()
            else:
                ppd, gtt, gtts, preds, mae, r2, mse, rte = solver.test()
                # preds, gtts = solver.test()
        elif config.mode == 'test':
            if config.test_vis:
                ppd, gtt, gtts, preds, mae, r2, mse, rte, filenames = solver.test()
            else:
                ppd, gtt, gtts, preds, mae, r2, mse, rte = solver.test()
        if config.mode == 'test':
            print('Checking testign data')
            images = data_vis(test_loader, config.notree, config.conf_score, config.use_color, config.add_chmInput, config.add_seg, add_activ_loss_descend = max(config.add_activ_loss_descend, config.add_activ_14reso), num = 1, model = solver.nnet, devi = solver.device, test = 1, test_vis = config.test_vis)
            # if config.showmap:
            #     map_vis(config, test_loader, config.notree, config.conf_score, config.use_color, num = 1, model = solver.nnet, devi = solver.device, test = 1)

    del train_loader, valid_loader, test_loader
    if 'test' in config.mode:
        if config.test_vis:
            return ppd, gtt, gtts, preds, mae, r2, mse, rte, filenames
        else:
            return ppd, gtt, gtts, preds, mae, r2, mse, rte
            # return preds, gtts
    else:
        return

def data_vis(loader, notree, conf_score, use_color, add_chm, add_seg, add_activ_loss_descend, num = 3, model = '', devi = '', test = 0, twoinput = 0, outputs = 0, test_vis = 0):
    # for _ in range(num):
    for itt, data in enumerate(loader):
        if itt < num:
            if not conf_score:
                images, labels = data
            else:

                if twoinput:

                    images, input2, labels, sp_wei = data

                elif outputs:
                    images, labels, add_op1, add_op2, sp_wei = data
                elif test_vis:
                    try:

                        images, labels, sp_wei, fn = data
                    except:
                        images, labels, chm_activ, sp_wei, fn = data

                # elif not test and add_activ_loss_descend:

                #     images, labels, chm_activ, sp_wei = next(dataiter)
                else:

                    # images, labels, chm_activ, sp_wei = data
                    images, labels, sp_wei = data
                # ipdb.set_trace()
                sp_wei = sp_wei.numpy()
            labels = labels.numpy()
            if twoinput:
                input2 = input2.numpy()
            if outputs:
                add_op1, add_op2 = add_op1.numpy(), add_op2.numpy()
            print('lb', labels.shape)
            print(images.shape)
            print(images.mean())
            print(images.std())
            print(images.max(), images.min())
            print(images[0, 0, :50, :50])
            if test:
                model.train(False)
                model.eval()
                images = images.to(devi)
                pre = model(images)
                pre = pre.cpu().detach().numpy()
                images = images.cpu().detach()

            try:
                if not add_chm:
                    if test:
                        plt.figure(figsize = (30,30)) # size(width, height)
                        for i in range(10):
                            for j in range(10): # column
                                plt.subplot(10, 10, 10*j + i+1)
                                curim = images[10*j + i].numpy()
                                curim = np.transpose(curim, axes=(1,2,0)) # Channel at the end

                                # recover from meanstd
                                imstd = np.array([0.229, 0.224, 0.225])
                                immean = np.array([0.485, 0.456, 0.406])
                                curim = curim*imstd + immean
                                # print(curim.max(), curim.min())
                                plt.imshow(curim[:, :, :3])
                                plt.xticks([])
                                plt.yticks([])
                                # plt.title(str(labels[i]))
                                if notree:
                                    plt.title([int(labels[10*j + i][0]), labels[10*j + i][1], labels[10*j + i][2]])
                                else:
                                    if conf_score:
                                        try:
                                            plt.title("NFI label: %.1f\n Prediction: %.1f"%(labels[10*j + i], pre[10*j + i]))
                                        except:
                                            plt.title("%s "%(fn[10*j + i]))

                                    else:
                                        plt.title(str(labels[10*j + i]))

                        plt.tight_layout()

                        plt.show(block = False)
                    else:
                        plt.figure(figsize = (20,6)) # size(width, height)
                        for i in range(6):

                            plt.subplot(1, 6, i+1)
                            curim = images[i].numpy()
                            curim = np.transpose(curim, axes=(1,2,0)) # Channel at the end

                            # recover from meanstd
                            imstd = np.array([0.229, 0.224, 0.225])
                            immean = np.array([0.485, 0.456, 0.406])
                            curim = curim*imstd + immean
                            # print(curim.max(), curim.min())
                            # ipdb.set_trace()

                            if curim.max()> 20 and curim.dtype != 'int8':
                                curim = curim.astype('int8')
                            plt.imshow(curim[:, :, :3])
                            # plt.title(str(labels[i]))
                            if notree:
                                plt.title([int(labels[i][0]), labels[i][1], labels[i][2]])
                            else:
                                if conf_score:
                                    if twoinput:
                                        plt.title("NFI label: %.1f\n Weight: %.1f \n Input2: %.1f "%(labels[i], sp_wei[i], input2[i]))
                                    elif outputs:
                                        plt.title("NFI label: %.1f\n Weight: %.1f \n addop1: %.2f \n addop2: %.2f "%(labels[i], sp_wei[i], add_op1[i], add_op2[i]))

                                    else:
                                        plt.title("NFI label: %.1f\n Weight: %.1f "%(labels[i], sp_wei[i]))

                                else:
                                    plt.title(str(labels[i]))

                        plt.tight_layout()
                        plt.show(block = True)


                elif add_chm:
                    if use_color:
                        plt.figure(figsize = (20,8)) # size(width, height)

                        if test:
                            plt.figure(figsize = (20,20)) # size(width, height)
                            for i in range(6):
                                for j in range(6): # column
                                    plt.subplot(6, 6, 6*j + i+1)
                                    curim = images[6*j + i].numpy()
                                    curim = np.transpose(curim, axes=(1,2,0)) # Channel at the end

                                    # recover from meanstd
                                    imstd = np.array([0.229, 0.224, 0.225])
                                    immean = np.array([0.485, 0.456, 0.406])
                                    curim = curim*imstd + immean
                                    # print(curim.max(), curim.min())
                                    plt.imshow(curim[:, :, 0], cmap = 'gray')
                                    # plt.title(str(labels[i]))
                                    if notree:
                                        plt.title([int(labels[6*j + i][0]), labels[6*j + i][1], labels[6*j + i][2]])
                                    else:
                                        if conf_score:
                                            try:
                                                plt.title("NFI label: %.1f\n Prediction: %.1f"%(labels[6*j + i], pre[6*j + i]))
                                            except:
                                                plt.title("label: %.1f;\n "%(labels[6*j + i]))

                                        else:
                                            plt.title(str(labels[6*j + i]))

                            plt.tight_layout()
                            plt.show(block = False)

                        else:

                            for i in range(6):

                                plt.subplot(3, 6, i+1)
                                curim = images[i].numpy()
                                curim = np.transpose(curim, axes=(1,2,0)) # Channel at the end
                                # recover from meanstd
                                imstd = np.array([0.229, 0.224, 0.225])
                                immean = np.array([0.485, 0.456, 0.406])
                                curim = curim*imstd + immean
                                # print(curim.shape)
                                # print(curim[:, :, :3].mean())
                                # import ipdb
                                # ipdb.set_trace()

                                plt.imshow(curim[:, :, 0].astype(float), cmap = 'gray')


                                if notree:
                                    plt.title([int(labels[i][0]), labels[i][1], labels[i][2]])
                                else:
                                    if conf_score:
                                        # plt.title(str(labels[i]) + ' sample weight: ' + str(sp_wei[i]))
                                        if test:
                                            plt.title("NFI label: %.1f\n Predidction: %.1f"%(labels[i], pre[i]))
                                        else:
                                            plt.title("NFI label: %.1f\nWeight: %.1f"%(labels[i], sp_wei[i]))
                                    else:
                                        plt.title(str(labels[i]))

                            for j in range(6):
                                # second row
                                # chm
                                plt.subplot(3, 6, j+7)
                                curim = images[j].numpy()
                                curim = np.transpose(curim, axes=(1,2,0)) # Channel at the end
                                imstd = np.array([0.229, 0.224, 0.225])
                                immean = np.array([0.485, 0.456, 0.406])
                                curim = curim*imstd + immean
                                plt.imshow(curim[:, :, 1], cmap = 'gray')
                                plt.colorbar()
                                # plt.title('max height: ' + str(round(curim[:, :, -1].max())))

                            for j in range(6):
                                # second row
                                # chm
                                plt.subplot(3, 6, j+13)
                                curim = images[j].numpy()
                                curim = np.transpose(curim, axes=(1,2,0)) # Channel at the end
                                imstd = np.array([0.229, 0.224, 0.225])
                                immean = np.array([0.485, 0.456, 0.406])
                                curim = curim*imstd + immean
                                plt.imshow(curim[:, :, -1], cmap = 'gray')
                                plt.colorbar()
                                # plt.title('max height: ' + str(round(curim[:, :, -1].max())))
                            plt.show(block = False)
                    else:
                        if add_seg:
                            plt.figure(figsize = (20,8)) # size(width, height)
                            for i in range(6):

                                plt.subplot(2, 6, i+1)
                                curim = images[i].numpy()
                                curim = np.transpose(curim, axes=(1,2,0)) # Channel at the end
                                imstd = np.array([0.229, 0.224, 0.225])
                                immean = np.array([0.485, 0.456, 0.406])
                                curim = curim*imstd + immean
                                # print(curim.shape)
                                # print(curim[:, :, 0].mean())
                                # import ipdb
                                # ipdb.set_trace()
                                plt.imshow(curim[:, :, 0].astype(int), cmap = 'gray')


                                if notree:
                                    plt.title([int(labels[i][0]), labels[i][1], labels[i][2]])
                                else:
                                    if conf_score:
                                        # plt.title(str(labels[i]) + ' sample weight: ' + str(sp_wei[i]))
                                        if test:
                                            plt.title("label: %.1f; weight: %.2f\n pred: %.1f"%(labels[i], sp_wei[i], pre[i]))
                                        else:
                                            plt.title("label: %.1f; weight: %.2f;\n Max H: %.1f"%(labels[i], sp_wei[i], curim[:, :, 0].max()))
                                    else:
                                        plt.title(str(labels[i]))
                            plt.colorbar()
                            for j in range(6):
                                # second row
                                # chm
                                plt.subplot(2, 6, j+7)
                                curim = images[j].numpy()
                                curim = np.transpose(curim, axes=(1,2,0)) # Channel at the end
                                plt.imshow(curim[:, :, -1]*curim[:,:,0], cmap = 'gray')
                            plt.show(block=False)

                        elif not add_seg:
                            plt.figure(figsize = (20,8)) # size(width, height)
                            for j in range(6):
                                # second row
                                # chm
                                plt.subplot(2, 6, j+1)
                                curim = images[j].numpy()
                                curim = np.transpose(curim, axes=(1,2,0)) # Channel at the end
                                plt.imshow(curim[:, :, -1], cmap = 'gray')

                                if notree:
                                    plt.title([int(labels[j][0]), labels[j][1], labels[j][2]])
                                else:
                                    if conf_score:
                                        # plt.title(str(labels[i]) + ' sample weight: ' + str(sp_wei[i]))
                                        if test:
                                            plt.title("label: %.1f; weight: %.2f\n pred: %.1f\n max H: %.1f"%(labels[j], sp_wei[j], pre[j], curim[:, :, -1].max()))
                                        else:
                                            plt.title("label: %.1f; weight: %.1f"%(labels[j], sp_wei[j]))
                                    else:
                                        plt.title(str(labels[j]))
                            plt.colorbar()
                            plt.show(block=False)
            except:
                continue
                print('skipping')
    if twoinput:
        return images, input2
    else:
        return images




def map_vis(config, loader, notree, conf_score, use_color, num = 3, model = '', devi = '', test = 0, twoinput = 0, outputs = 0):
    for _ in range(num):
        dataiter = iter(loader)

        if not conf_score:
            # images, labels = dataiter.next()
            images, labels = next(dataiter)
        else:

            if twoinput:

                images, input2, labels, sp_wei = next(dataiter)

            elif outputs:
                images, labels, add_op1, add_op2, sp_wei = next(dataiter)
            else:
                if config.test_vis:
                    images, labels, sp_wei, filename = next(dataiter)
                else:
                    images, labels, sp_wei = next(dataiter)

            sp_wei = sp_wei.numpy()
        labels = labels.numpy()
        if twoinput:
            input2 = input2.numpy()
        if outputs:
            add_op1, add_op2 = add_op1.numpy(), add_op2.numpy()

        def get_features(name):
            def hook(model, input, output):
                features[name] = output.detach()
            return hook

        try:
            model.features[-1].register_forward_hook(get_features('feats'))
        except:
            model.trunk_output[-1].register_forward_hook(get_features('feats'))



        if test:
            model.train(False)
            model.eval()
            images = images.to(devi)
            FEATS = []

            # placeholder for batch features
            features = {}

            for im in images:
                im = im.unsqueeze(0)
                pre = model(im)
                pre = pre.cpu().detach().numpy()
                im = im.cpu().detach()

                FEATS.append(features['feats'].cpu().numpy())


            FEATS = np.concatenate(FEATS)

            print('- feats shape:', FEATS.shape)

            gray_scale = np.sum(FEATS, axis = (1))
            gray_scale = gray_scale / FEATS.shape[1]
            print(gray_scale.shape)
            # ipdb.set_trace()
            if not os.path.exists(config.act_result_path):
                os.mkdir(config.act_result_path)

            for idd in range(len(gray_scale)):

                savename = os.path.join(config.act_result_path, filename[idd] + '_act.tif')
                imsave(savename,gray_scale[idd])
                # plt.imsave(savename, gray_scale[idd], cmap='gray')
        # visualize
        # plt.figure(figsize = (20,20)) # size(width, height)
        # # fig, axes = plt.subplots(nrows=6, ncols=6)
        # for i in range(6):
        #     for j in range(6): # column
        #         plt.subplot(6, 6, 6*j + i+1)
        #         curim = gray_scale[6*j + i]
        #         # curim = np.transpose(curim, axes=(1,2,0)) # Channel at the end
        #         # print(curim.max(), curim.min())
        #         plt.imshow(curim[:, :])
        #         # plt.title(str(labels[i]))
        #         if notree:
        #             plt.title([int(labels[6*j + i][0]), labels[6*j + i][1], labels[6*j + i][2]])
        #         else:
        #             if conf_score:
        #                 try:
        #                     plt.title("NFI label: %.1f\n Prediction: %.1f"%(labels[6*j + i], pre[6*j + i]))
        #                 except:
        #                     plt.title("label: %.1f;\n "%(labels[6*j + i]))
        #
        #             else:
        #                 plt.title(str(labels[6*j + i]))
        # fig.colorbar(im, ax=axes.ravel().tolist())
        # fig.subplots_adjust(right=0.8)
        # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        # fig.colorbar(im, cax=cbar_ax)
        plt.tight_layout()

        plt.show(block=False)

    if twoinput:
        return images, input2
    else:
        return images





    ##############################################################################################


    #################################################################################################################
