import numpy as np
from sklearn.model_selection import train_test_split as tts
import torch.utils.data as data
import torch
from CDIPutils import SQUIDify

import os
import random


class DataLoader:

    """
    Loads the data one file at a time during the training process to avoid
    having all the data in memory at once. Also does the pre-processing
    including the squid angle conversion, normalisation and adding the noise.

    Input:
        - self
        - config (object): The config class contains all of the important
            (hyper)parameters needed during training.
        - file_num (int): Indicates which file needs to be loaded.
    """

    def __init__(self, config, file_num):
        # Setting seeds
        self.config = config
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        random.seed(self.config.seed)

        # Fetching the data from the files
        train_in_file = os.path.join(self.config.data_dir,
            self.config.train_input+str(file_num).zfill(3) +
            '.npy')
        train_out_file = os.path.join(self.config.data_dir,
            self.config.train_output+str(file_num).zfill(3) +
            '.npy')

        # Loading the data into a NumPy array
        inputData = np.load(train_in_file)
        outputData = np.load(train_out_file)

        # inputData = torch.from_numpy(inputData.swapaxes(1,3).astype('float32')).cuda(self.config.deviceName)
        # outputData = torch.from_numpy(outputData.swapaxes(1,3).astype('float32')).cuda(self.config.deviceName)
        
        # Either converting the raw data using SQUIDify or turing the squidtools
        # image 90 degrees
        if self.config.SQUIDify:
            inputData = SQUIDify(inputData, self.config.SQUIDAngle, self.config.SQUIDOrient)
        else:
            inputData = np.flip(inputData, axis=2).swapaxes(1, 2)

        # Creating a torch tensor from the numpy arrays and putting them on the
        # GPU
        inputData = torch.from_numpy(inputData.swapaxes(1,3).astype('float32')).cuda(self.config.deviceName)
        outputData = torch.from_numpy(outputData.swapaxes(1,3).astype('float32')).cuda(self.config.deviceName)

        # Normalising the data
        self.inputData = (inputData-inputData.mean()) / inputData.std()
        self.outputData = (outputData-outputData.mean()) / outputData.std()

        # Adding Gaussian noise if possible
        if self.config.noise:
            noise = torch.normal(mean=self.inputData.mean(), std=self.config.noiseStd,
                                 size=np.shape(self.inputData)).cuda(self.config.deviceName)
            self.inputData = self.inputData + noise

        # self.inputData = (inputData-inputData.mean()) / inputData.std()
        # self.outputData = (outputData-outputData.mean()) / outputData.std()

        # Splitting the data randomly into training, validation and testing
        # sets using a set seed
        self.inTrainSet, inValTest, self.outTrainSet, outValTest = tts(
            self.inputData, self.outputData, test_size=self.config.val_test_size,
            random_state=self.config.seed)
        self.inValSet, self.inTestSet, self.outValSet, self.outTestSet = tts(
            inValTest, outValTest, test_size=0.5,
            random_state=self.config.seed)
        
        # Finding the length of the array
        self.len = self.inTrainSet.shape[0]


    def trainLoader(self, batch_size):
        """
        Loads the training set.

        Input:
            - self
            - batch_size (int): How many images per batch
        
        Ouptut:
            - dataLoader (torch.utils.data.DataLoader): Torch DataLoader object
        """

        idx = np.random.choice(self.len, batch_size)
        trainSet = data.TensorDataset(self.inTrainSet[idx], self.outTrainSet[idx])
        dataLoader = data.DataLoader(trainSet, batch_size=batch_size,
                                     shuffle=True)
        return dataLoader

    def val(self):
        """Loads validation set"""

        return self.inValSet, self.outValSet

    def test(self):
        """Loads testing set"""
        return self.inTestSet, self.outTestSet


if __name__ == "__main__":

    class config:
        data_dir = "/local/s4451856/SQUID-CNN/training_data"
        train_input = "64x64_SQUID_B_"
        train_output = "64x64_50um_Jxy_"
        seed = 42

        in_im_c = 1
        out_im_c = 2

    loader = DataLoader(config, 1) 
    print(loader.test())