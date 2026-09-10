import numpy as np
import torch
import copy

from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.plots import plot_convergence

from TorchUnet import HPOUNet
from dataLoader import DataLoader
from training import Trainer

class BO:

    """
    Does Hyperparameter Optimisation (HPO) using Bayesian Optimisation (BO).

    Input:
        - self
        - config (object): The config class contains all of the important
            (hyper)parameters needed during training.
    """
    
    def __init__(self, config):
        # Setting config, bestResults as infinite (lower MSE is better), best
        # hypereparamters and the seed.
        self.config = config
        self.bestResults = [np.inf]
        self.bestHP = []
        np.random.seed(self.config.seed)

        # Setting up the search space with the range of possible hyperparameters
        self.searchSpace = [
            Integer(3, 6, name="numUpLayers"),
            Integer(4, 64, name="batchSize"),
            Real(0.000001, 0.1, name="lr"),
            Integer(5, 150, name="epochs"),
            Real(0.0001, 1, name="noiseStd")
        ]

        # Running the BO process using gp_minimize()
        self.BOresults = gp_minimize(
            self.objectiveFunction,
            self.searchSpace,
            n_calls=self.config.num_obFuncCalls,
            random_state=self.config.seed
        )

        # Saving and printing the best result after the process is done
        torch.save(self.bestModel.state_dict(), config.checkpoint_dir+'model.pt')
        print("Best Results:", self.bestResults)
        print("Best Hyperparams:", self.bestHP)

    def objectiveFunction(self, hp):

        """
        The function that is called to see how good each guess was.

        Input:
            - self
            - hp (list): The list of hyperparameters chosen from the search space
        """

        # Setting variables for each hyperparameter
        numUpLayers = hp[0]
        batchSize = 64
        lr = hp[2]
        epochs = hp[3]
        noiseStd = hp[4]
        self.config.obFuncCall += 1

        # Setting up the layers of the model
        self.config.layers = []
        thisLayer = self.config.im_h
        for upLayer in range(1, numUpLayers+1):

            self.config.layers.append(thisLayer)
            thisLayer = thisLayer * 2

        # Setting the hyperparameters in the config object
        self.config.batch_size = batchSize
        self.config.learning_rate = lr
        self.config.num_epochs = epochs
        self.config.noiseStd = noiseStd

        # Printing the parameters
        print(f"Current parameters: numUpLayers: {numUpLayers},",
              f"totalLayers: {self.config.layers},", 
              f"batchSize: {batchSize},",
              f"lr: {lr},",
              f"epochs: {epochs},",
              f"noiseStd: {noiseStd}"
              )

        # Setting up the model, optimiser and trainer for training
        model = HPOUNet(config).cuda(self.config.deviceName)
        optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        trainer = Trainer(model, optimiser, DataLoader, config)
        trainedModel = trainer.train()
        # torch.save(model.state_dict(), config.checkpoint_dir+'model.pt')

        # Measuring how well the model did
        results = trainer.measure_accuracy()

        # Saving the results if they were the best so far
        if results[0] < self.bestResults[0]:
            self.bestResults = copy.copy(results)
            self.bestHP = hp
            self.bestModel = copy.copy(trainedModel)

        # Printing the results
        print(f"Last val loss: {round(results[0], 3)},",
                f"Last val SSIM: {round(results[1], 2)}",
                f"Last test loss: {round(results[2], 3)},",
                f"Last test SSIM: {round(results[3], 2)}")

        # Returing the validation loss
        return results[0]
        

if __name__ == "__main__":

    class config(object):

        # Model hyperparameters
        im_h = 64 # Image height
        im_w = 64 # Image width
        in_im_c = 1 # Number of components in each pixel in the input image
        out_im_c = 2 # Number of components in each pixel in the input image
        avgPool = False # If you want average pooling layers, uses max pooling if False

        # Training Hyperparameters
        SQUIDify = False # Set to True for angle conversion, set to False for squidtools data
        SQUIDAngle = 62 # Angle of the SQUID (Only relevant for SQUIDify)
        SQUIDOrient = 0 # Orientation of the SQUID (0 for xz, 1 for yz) (Only relevant for SQUIDify)

        # layers = [64, 128, 256, 512] # Size and shape of the network (Only represents the downsampling side)

        # batch_size = 64 # How big the batch should be for each training step
        seed = 42 # Random seed (Must have or train and val sets will be mixed)
        loss = torch.nn.MSELoss() # None = self.custom_mse_loss
        # learning_rate = 1e-4
        val_test_size = 0.1 # Size of the validation and testing function each

        noise = True # Add noise?
        # noiseStd = 0.1 # Standard deviation of the Gaussian noise

        # num_epochs = 50 # How many epochs per training run?
        num_files = 77 # How many files of training data?
        num_obFuncCalls = 20 # How many object function calls before stopping?
        obFuncCall = 0 # Counter for the Object Function, don't change.

        checkpoint_dir = "/local/s4451856/SQUID-CNN/CDIP/models/CDIP_64_50um_SQUID_HPO_noise/" # Directory to save the model in
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

    BO(config)
    # torch.save(bestModel.state_dict(), config.checkpoint_dir+'model.pt')
    # print("Best Results:", bestResults)
    # print("Best HP:", bestHP)
    # print("LastResults:", latestResults)