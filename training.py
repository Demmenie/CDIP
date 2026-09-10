import tqdm
import numpy as np
import torch
import copy

from dataLoader import DataLoader
from TorchUnet import UNet, HPOUNet
from CDIPutils import SQUIDify

from skimage.io import imread

from fast_ssim import ssim

# import tensorflow as tf


class Trainer:

    """
    Trains and evaluates the model.

    Input:
        - self
        - model (object): Either a UNet or HPOUNet model.
        - optimiser (torch.optim object): An optimiser from torch.
        - dataLoader (object): The DataLoader class from dataLoader.py
        - config (object): The config class contains all of the important
            (hyper)parameters needed during training.
    """

    def __init__(self, model, optimiser, dataLoader, config):
        # Setting variables
        self.config = config
        self.config.device = torch.device(self.config.deviceName)
        self.model = model#.cuda(self.config.device)
        self.optimiser = optimiser
        self.dataLoader = dataLoader
    
        # Setting seed
        torch.manual_seed(self.config.seed)

        # Setting loss function if there isn't one specified
        if not self.config.loss:
            self.config.loss = self.custom_mse_loss

        # Getting the validation and test set
        self.valSetIn, self.valSetOut, self.testSetIn, self.testSetOut = self.getValTestSet()

        # self.valMetric = SSIM(data_range=self.valSetIn.max() - self.valSetIn.min(),
        #                       device=torch.device("cuda:0"))
        # self.testMetric = SSIM(data_range=self.testSetIn.max() - self.testSetIn.min(),
        #                       device=torch.device("cuda:0"))
        

    def getValTestSet(self):
        """
        Iterates through the data files, collects and returns the validation and
        testing sets

        Input:
            - self

        Output:
            - valSet (numpy.Array)
            - testSet (numpy.Array)
        """

        print("Getting validation and testing data...")

        valSetIn = torch.Tensor(np.ndarray([0, self.config.in_im_c, self.config.im_h, self.config.im_w])).cuda(self.config.device)
        valSetOut = torch.Tensor(np.ndarray([0, self.config.out_im_c, self.config.im_h, self.config.im_w])).cuda(self.config.device)
        testSetIn = torch.Tensor(np.ndarray([0, self.config.in_im_c, self.config.im_h, self.config.im_w])).cuda(self.config.device)
        testSetOut = torch.Tensor(np.ndarray([0, self.config.out_im_c, self.config.im_h, self.config.im_w])).cuda(self.config.device)

        progressBar = tqdm.tqdm(total=self.config.num_files,
            desc='Files', position=0)

        for file_num in range(self.config.num_files):
            dataLoader = self.dataLoader(self.config, file_num)
            
            valIn, valOut = dataLoader.val()
            # print(valSetIn.shape, valIn.shape)
            valSetIn = torch.concatenate((valSetIn, valIn), axis=0)
            valSetOut = torch.concatenate((valSetOut, valOut), axis=0)

            testIn, testOut = dataLoader.test()
            testSetIn = torch.concatenate((testSetIn, testIn), axis=0)
            testSetOut = torch.concatenate((testSetOut, testOut), axis=0)

            # Updating the progress bar
            progressBar.update()

        # valSetIn = torch.from_numpy(valSetIn.swapaxes(1,3).astype('float32')).cuda(self.config.device)
        # valSetOut = torch.from_numpy(valSetOut.swapaxes(1,3).astype('float32')).cuda(self.config.device)

        # testSetIn = torch.from_numpy(testSetIn.swapaxes(1,3).astype('float32')).cuda(self.config.device)
        # testSetOut = torch.from_numpy(testSetOut.swapaxes(1,3).astype('float32')).cuda(self.config.device)

        return valSetIn, valSetOut, testSetIn, testSetOut


    def measure_accuracy(self):

        """
        Measures the accuracy of the model after training.

        Input:
            - self

        Output:
            - avValLoss (float): The average Validation loss
            - avValSSIM (float): The average Validation SSIM
            - avTestLoss (float): The average Testing loss
            - avTestSSIM (float): The average Testing SSIM
        """

        # Because the validation and test sets are so large, we have to do
        # prediction in batches so that the network and data fit into memory
        valSetLength = len(self.valSetIn)
        valDataRange = self.valSetIn.max() - self.valSetOut.min()
        testDataRange = self.testSetIn.max() - self.testSetOut.min()
    
        # Predicting validation set
        valLossList = []
        valSSIMList = []
        testLossList = []
        testSSIMList = []

        # Iterating through different noise levels to get an average over all
        for noise in [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1]:

            # Setting variables
            self.config.noiseStd = noise
            self.valSetIn, self.valSetOut, self.testSetIn, self.testSetOut = [], [], [], []
            self.valSetIn, self.valSetOut, self.testSetIn, self.testSetOut = self.getValTestSet()
            print(f"Measuring accuracy at {self.config.noiseStd} noise Std...")

            # Creating a progress bar for each setting
            progressBar = tqdm.tqdm(total=valSetLength // 
            self.config.batch_size, desc='Measuring Accuracy', position=0)
            for batchIndex in range(0, valSetLength, self.config.batch_size):

                # Getting the validation batch
                valSetInBatch = self.valSetIn[batchIndex: (batchIndex +
                    self.config.batch_size)]
                valSetOutBatch = self.valSetOut[batchIndex: (batchIndex +
                    self.config.batch_size)]
                
                # Getting the testing batch
                testSetInBatch = self.testSetIn[batchIndex: (batchIndex +
                    self.config.batch_size)]
                testSetOutBatch = self.testSetOut[batchIndex: (batchIndex +
                    self.config.batch_size)]

                
                # Performing Inference
                with torch.inference_mode():
                    valPred = self.model(valSetInBatch)
                    testPred = self.model(testSetInBatch)

   
                # Finding the loss and adding it to the loss history list
                val_loss = self.config.loss(valSetOutBatch, valPred).item()
                valLossList.append(val_loss)

                test_loss = self.config.loss(testSetOutBatch, testPred).item()
                testLossList.append(test_loss)

                # Getting the predictions back from GPU
                valPred = valPred.cpu().numpy().swapaxes(1,3)
                valSetOutBatch = valSetOutBatch.cpu().numpy().swapaxes(1,3)

                testPred = testPred.cpu().numpy().swapaxes(1,3)
                testSetOutBatch = testSetOutBatch.cpu().numpy().swapaxes(1,3)


                # Appending results to results list
                for index in range(len(valSetOutBatch)):
                    valSSIM = ssim(valSetOutBatch[index], valPred[index],
                                data_range=valDataRange)
                    valSSIMList.append(valSSIM)

                    testSSIM = ssim(testSetOutBatch[index], testPred[index],
                                data_range=testDataRange)
                    testSSIMList.append(testSSIM)

                    progressBar.update()

        # Averaging metrics
        avValLoss = np.mean(valLossList)
        avValSSIM = np.mean(valSSIMList)
        avTestLoss = np.mean(testLossList)
        avTestSSIM = np.mean(testSSIMList)

        return avValLoss, avValSSIM, avTestLoss, avTestSSIM


    def train(self):
        """
        This function iterates through the training process according to the
        configuration specified.

        Input:
            - self

        Output:
            - model
        """

        # Specifying loss history and training log (Not often used because this
        # lengthens the process significantly)

        # loss_history = np.ndarray((3, self.config.num_epochs))
        # SSIM_history = np.zeros((3, self.config.num_epochs))
        # valLoss, valSSIM, testLoss, testSSIM = (0, 0, 0, 0)

        # Iterating through each file for each epoch
        for epoch in range(self.config.num_epochs):

            trainLossHistory = []

            # Creating a progress bar for this file
            progressBar = tqdm.tqdm(total=self.config.num_files, desc='Files', position=0)                
            train_status = tqdm.tqdm(total=0, bar_format='{desc}',
                position=1)
            for file_num in range(0, self.config.num_files):
                
                # Getting the data for this file
                self.data = DataLoader(self.config, file_num)


                # Iterating through training steps for each data batch
                trainLoader = self.data.trainLoader(self.config.batch_size)
                for inputData, targetData in trainLoader:

                    # Getting the prediction & loss and optimising the model
                    self.optimiser.zero_grad()
                    output = self.model(inputData)
                    loss = self.config.loss(output, targetData)
                    loss.backward()
                    self.optimiser.step()
                    trainLoss = loss.item()

                    trainLossHistory.append(trainLoss)

                # Giving a string for the progress bar
                train_status.set_description_str(
                    f"File: {file_num} Loss: {trainLoss}")

                # Updating the progress bar
                progressBar.update()

            meanTrainLoss = np.mean(trainLossHistory)
            #valLoss, valSSIM, testLoss, testSSIM = self.measure_accuracy()
            
            print(f"Epoch: {epoch}/{self.config.num_epochs},",
                f"ObFunc Call: {self.config.obFuncCall},", 
                f"Training Loss: {round(meanTrainLoss, 3)},",
                f"Layers: {self.config.layers},",
                f"Batch Size: {self.config.batch_size},",
                f"Learning Rate: {self.config.learning_rate}",
                f"noise: {self.config.noiseStd}")
                #f"Training SSIM: {},",
                # f"Last val loss: {round(valLoss, 3)},",
                # f"Last val SSIM: {round(valSSIM, 2)}",
                # f"Last test loss: {round(testLoss, 3)},",
                # f"Last test SSIM: {round(testSSIM, 2)}")

            # Saving loss history at the end of each file
            # loss_history[0, epoch] = trainLoss
            # loss_history[1, epoch] = valLoss
            # loss_history[2, epoch] = testLoss

            # # SSIM_history[0, epoch] = trainSSIM
            # SSIM_history[1, epoch] = valSSIM
            # SSIM_history[2, epoch] = testSSIM

            # Saving the model at the end of each epoch
            # torch.save(self.model.state_dict(), self.config.checkpoint_dir+'model.pt')

            # loss_history.tofile(self.config.checkpoint_dir+"loss_history.npy")
            # SSIM_history.tofile(self.config.checkpoint_dir+"SSIM_history.npy")
            
        return self.model


if __name__ == "__main__":
    class config(object):

        # Model hyperparameters
        im_h = 64 # Image height
        im_w = 64 # Image width
        in_im_c = 1 # Number of components in each pixel in the input image
        out_im_c = 2 # Number of components in each pixel in the input image
        avgPool = False

        # Training Hyperparameters        
        SQUIDify = False # Set to True for angle conversion, set to False for squidtools data
        SQUIDAngle = 62 # Angle of the SQUID (Only relevant for SQUIDify)
        SQUIDOrient = 0 # Orientation of the SQUID (0 for xz, 1 for yz) (Only relevant for SQUIDify)

        layers = [64, 128, 256, 512] # Size and shape of the network (Only represents the downsampling side)

        batch_size = 64 # How big the batch should be for each training step
        seed = 42 # Random seed (Must have or train and val sets will be mixed)
        loss = torch.nn.MSELoss() # None = self.custom_mse_loss
        learning_rate = 1e-4
        val_test_size = 0.1 # Size of the validation and testing function each

        noise = True # Add noise?
        noiseStd = 0.1 # Standard deviation of the Gaussian noise
        # noiseStdRange = (0.00001, 0.0005)

        num_epochs = 25 # How many epochs per training run?
        num_files = 77 # How many files of training data?

        obFuncCall = 0 # Counter for the Object Function, don't change.

        checkpoint_dir = "/local/s4451856/SQUID-CNN/CDIP/models/CDIP_SQUID_B_noise/" # Directory to save the model in
        data_dir = "/data/s4451856/training_data" # Training data directory
        train_input = "64x64_SQUID_B_" # Name of the input data files
        train_output = "64x64_50um_Jxy_" # Name of the label data files
        # test_input = "64x64_50um_Bxyz_000.npy"
        # test_output = "64x64_50um_Jxy_000.npy"

        deviceName = "cuda:1" # Name of the GPU / Device you want to use


    # print(config.__dict__)
    # config.loss = str(type(config.loss))
    # json.dump(dict(config.__dict__), open(
    #     config.checkpoint_dir+"config.json", "w"))

    config.device = torch.device(config.deviceName)
    model = HPOUNet(config).cuda(config.device)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    trainer = Trainer(model, optimiser, DataLoader, config)
    trainedModel = trainer.train()
    torch.save(trainedModel.state_dict(), config.checkpoint_dir+'model.pt')

    results = trainer.measure_accuracy()
    print(f"Val Loss: {results[0]}, Val SSIM: {results[1]},",
          f"Test Loss: {results[2]}, Test SSIM: {results[3]}")
