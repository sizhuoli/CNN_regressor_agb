##  Image regression / classification

----------------------------------------

### Set up conda enviroment

```
conda env create -f environment_pytorch_cnn_regressor.yml

```

----


### Train a regression model with regression head design A (fully connect layers after global pooling)

```

python main_training.py --head A

```
#### Set configs in conf/config_training_regression_head_A-dense-design.yaml

----


### Train a regression model with regression head design B (activation map)

```

python main_training.py --head B

```
#### Set configs in conf/config_training_regression-head-B-map-design.yaml

----




#### Data format

Each image file is saved as a tiff located in dataset/train/ or dataset/valid/ or dataset/test/

Target label of each image file is saved in a csv

Linking image file with corresponding taget label: 

Each image file is given an unique file name which points to a row in the csv file

SEE example csv file in dataset/...csv

E.g. image tiff file may have a file name '6531C.tif'


----------------------------------------

### Large scale inference using a trained regression model

```

python inference_run.py

```

#### Set configs in conf/config_inference_rgb.yaml

----------------------------------------

### Train a classification model

```

python main_classification.py

```

#### Set configs in conf/config_classification.yaml
