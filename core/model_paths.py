# save model path and model type as a dict
model_paths_rgb = {
    'torchEfficientnetb0_dense': '/home/sizhuo/Desktop/code_repository/CNN_AGB/saved_models/old_data_new_train/rgb_recap/AGB_20221005-1623-torchEfficientnetb0_dense-Epo400-yearAft_2015-patch_224-Loss_L1-bands_RGB-_evened_val_test/bestR2.pkl',
    'torchEfficientnetb0_map':'/home/sizhuo/Desktop/code_repository/CNN_AGB/saved_models/old_data_new_train/rgb_recap/AGB_20230419-1123-torchEfficientnetb0_map-Epo300-yearAft_2015-patch_224-Loss_L1-bands_RGB-_new_rgb_models_relu_enforce_nonnegative/bestR2.pt',
    'torchEfficientnetb1_dense':'/home/sizhuo/Desktop/code_repository/CNN_AGB/saved_models/old_data_new_train/rgb_recap/AGB_20230823-2136-torchEfficientnetb1_dense-Epo300-yearAft_2015-patch_224-Loss_L1-bands_RGB-_added_model/bestR2.pt',
    'torchEfficientnetb1_map': '/home/sizhuo/Desktop/code_repository/CNN_AGB/saved_models/old_data_new_train/rgb_recap/AGB_20230418-2321-torchEfficientnetb1_map-Epo300-yearAft_2015-patch_224-Loss_L1-bands_RGB-_new_rgb_models_relu_enforce_nonnegative/bestR2.pt',
    'torchEfficientnetb2_dense': '/home/sizhuo/Desktop/code_repository/CNN_AGB/saved_models/old_data_new_train/rgb_recap/AGB_20230419-1134-torchEfficientnetb2_dense-Epo300-yearAft_2015-patch_224-Loss_L1-bands_RGB-_new_rgb_models/bestR2.pt',
    'torchEfficientnetb2_map': '/home/sizhuo/Desktop/code_repository/CNN_AGB/saved_models/old_data_new_train/rgb_recap/AGB_20230418-2323-torchEfficientnetb2_map-Epo300-yearAft_2015-patch_224-Loss_L1-bands_RGB-_new_rgb_models_relu_enforce_nonnegative/bestR2.pt',
    'regnetY800MF_dense': '/home/sizhuo/Desktop/code_repository/CNN_AGB/saved_models/old_data_new_train/rgb_recap/AGB_20230418-1157-regnetY800MF_dense-Epo300-yearAft_2015-patch_224-Loss_L1-bands_RGB-_new_rgb_models/bestR2.pt',
    'regnetY800MF_map': '/home/sizhuo/Desktop/code_repository/CNN_AGB/saved_models/old_data_new_train/rgb_recap/AGB_20230418-2314-regnetY800MF_map-Epo300-yearAft_2015-patch_224-Loss_L1-bands_RGB-_new_rgb_models_relu_enforce_nonnegative/bestR2.pt'
}


def prepare_saved_model(model_type, model_dict = model_paths_rgb):
    model_path = model_dict[model_type]
    return model_path



