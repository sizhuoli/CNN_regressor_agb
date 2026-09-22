#  Image-patch-level regression

A CNN model for forest biomass estimation from sub-meter resolution images

## Publication

Deep learning tree and forest biomass from sub-meter resolution optical imagery

Remote Sensing of Environment, Volume 345, 2026.

Sizhuo Li, Martin Brandt, Xiaoye Tong, Stefan Oehmcke, Christian Igel, Florian Reiner, Fabian Gieseke, Thomas Nord-Larsen, Rasmus Fensholt, Jerome Chave, Philippe Ciais.

https://www.sciencedirect.com/science/article/abs/pii/S0034425726003093

## The CNN model: training, evaluation & large-scale inference
### 0. Set up conda enviroment

```
conda env create -f environment_pytorch_cnn_regressor.yml

```



### 1.0 Train a regression model with regression head design A (fully connect layers after global pooling)

```

python main_training.py --head A

```
#### 1.0.0 Set configs in conf/config_training_regression_head_A-dense-design.yaml



### 1.1 Train a regression model with regression head design B (activation map)

```

python main_training.py --head B

```
#### 1.1.0 Set configs in conf/config_training_regression-head-B-map-design.yaml

----




#### 1.2 Data format

Each image file is saved as a tiff located in dataset/train/ or dataset/valid/ or dataset/test/

Target label of each image file is saved in a csv

Linking image file with corresponding taget label: 

Each image file is given an unique file name which points to a row in the csv file

SEE example csv file in dataset/...csv

E.g. image tiff file may have a file name '6531C.tif'


----------------------------------------

### 2. Evaluate trained model on the test data

```
python test_scores.py
```

#### 2.0 Set configs in conf/config_testscore_rgb.yaml

For evaluating models trained using only RGB bands

* Default settings for map design (design B)

* Need to change model architecture and add_... in configs for dense design (design A)


----


### 3. Large scale inference using a trained regression model

```

python inference_run.py

```

#### 3.0 Set configs in conf/config_inference_rgb.yaml


