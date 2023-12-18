##  Image regression / classification

----------------------------------------

### Train a regression model

```

python main_test.py

```


#### Set configs in conf/config_old_data.yaml

#### Data format

Each image file is saved as a tiff located in dataset/train/ or dataset/valid/ or dataset/test/

Target label of each image file is saved in a csv

Linking image file with corresponding taget label: 

Each image file is given an unique file name which points to a row in the csv file

SEE example csv file in dataset/...csv

E.g. image tiff file may have a file name '6531C.tif'


----------------------------------------

### Test a regression model

```

python test_scores.py

```

#### Set configs in conf/config_testscore_old_data.yaml

----------------------------------------

### Train a classification model

```

python main_classification.py

```

#### Set configs in conf/config_classification.yaml



