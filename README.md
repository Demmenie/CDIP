# Code for: Current Density Image Prediction (CDIP) from SQUID-on-tip Magnetic Field Data

## Overview

This repository contains the code and associated resources used in the Master's Thesis titled **"Current Density Image Prediction (CDIP) from SQUID-on-tip Magnetic Field Data"**, including:

- **UNet Code** - `TFUnet.py` contains a class for a unet made with tensorflow, while `TorchUnet.py` contains two classes made using Torch, one static and one which can be adjusted during HPO.
- **Training Code** - `training.py` Contains code to train and evaluate the model.
- **HPO code** - `hpo.py` Can find optimal hyperparameters using a Gaussian minimisation Bayesian Optimisation process.
- **Inference scripts** - The "inference" directory contains some jupyter notebooks in which you can do inference on images or whole datasets and measure performance
- **Data generation scripts** - The "data_generation" directory contains scripts to convert data into squidtools data.


## Getting started

### Prerequisites
Ensure that you have the following:
- Python 3.8 or higher
- Required python libraries (listed in `requirements.txt`)

### Installation

1. Download repository
2. Download one of the training datasets from: https://doi.org/10.7910/DVN/SD6PVP

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run `squidConverter.py` (Check the input and output directories are correct)
    ```bash
    cd data_generation
    python3 squidConverter.py
    ```

---


## Performing Inference

1. Open the jupyter notebook you want to run
2. Select the Python version you want to use
3. Run the cells with the model you want to test


## Training new models

1. Prepare your training data (If you're training a squidtools model make sure you have converted the data)
2. Check the config object at the bottom of `training.py` and modify according to the parameters you want
3. Run the training script:
    ```bash
    python3 training.py
    ```
3. Monitor the training process:
    - If the training loss isn't around 0.3 by the end of the first epoch then the hyperparameters are probably bad.
    - The model is saved in the `/models/` directory


### Configuration

You can change the configuration easily in the config object at the bottom of the `training.py` and `hpo.py` files, they are explained in the following list:

**Model Hyperparameters**
- `im_h` - Input/Output image height
- `im_w` - Input/Output image width
- `in_im_c` - Input image components (Usually either 3 for xyz or 1 for SQUID images)
- `out_im_c` - Output image components (Usually 2 for xy)
- `avgPool` - True for average pooling layers, False for Max Pooling layers

**Training Hyperparameters**
- `SQUIDify` - Set to True for angle conversion, set to False for squidtools data
- `SQUIDAngle` - The angle of the SQUID (Only relevant for SQUIDify)
- `SQUIDOrient` - The orientation of the SQUID (0 for xz, 1 for yz) (Only relevant for SQUIDify)

- `layers` - Size and shape of the network (Only represents the downsampling side) (Only relevant for training, not for hpo)
- `batch_size` - How big the batch should be for each training step (Only relevant for training, not for hpo)
- `seed` - Random seed for replication purposes (You must have a seed, otherwise training and validation sets will be mixed)
- `loss` - The loss function you want to use
- `learning_rate` - Learning rate for the model, usually 1e-4 works well (Only relevant for training, not for hpo)
- `val_test_size` - sets the size of the validation and testing sets, usually set at 0.1, so they're 10% each and training set is 80%.
- `noise` - Boolean indicator for noise
- `noiseStd` - Standard deviation for Gaussian noise (Only relevant for training, not for hpo)
- `num_epochs` - The number of epochs to do during training (Only relevant for training, not for hpo)
- `num_files` - How many files of data there are to train on
- `num_obFuncCalls` - How many object function calls the BO process should do before stopping (Only relevant for hpo, not for training alone)
- `obFuncCall` - Variable for the object function, don't change
- `checkpoint_dir` - Directory where the model will be stored
- `data_dir` - Directory where the data is stored
- `train_input` - Name of the input data files
- `train_output` - name of the label data files
- `deviceName` - Name of the device you want to train on

**`squidConverter.py`**
- `rho1` - The inner diameter of the SQUID
- `rho2` - The outer diameter of the SQUID
- `height` - The height of the SQUID above the sample
- `phi` - The angle of the SQUID relative to the sample

## Training Datasets

The following datasets are available for training:

- **50 micron and 64x64 resolution**: [https://doi.org/10.7910/DVN/QPCS0I](https://doi.org/10.7910/DVN/QPCS0I)
- **50 micron and 256x256 resolution**: [https://doi.org/10.7910/DVN/OPEX5N](https://doi.org/10.7910/DVN/OPEX5N)
- **500 micron and 64x64 resolution**: [https://doi.org/10.7910/DVN/SJDS2O](https://doi.org/10.7910/DVN/SJDS2O)

Note that while these datasets were generated from the same data distribution as the data used to train the models, none of the included configurations were used in our training set. Thefore, all data in these collections can also serve as validation data to our trained networks.

