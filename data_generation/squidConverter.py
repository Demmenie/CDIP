import numpy as np
import torch
import os
import tqdm

from squidtools.Forward import Forward2DCurrent, Forward2DParameters

class converter:

    """
    Converts the training data into SQUID data using squidtools (SOTtools)

    Input:
        - self
        - config (object): The config class contains all of the important
            (hyper)parameters needed during training.
    """

    def __init__(self, config):
        self.config = config

        # Initialising squidtools
        print("Initialising...")
        params = Forward2DParameters(
            10.0,
            10.0,
            64,
            64,
            config.rho1,
            config.rho2,
            config.height,
            config.phi,
            False,
            False
        )
        self.forward = Forward2DCurrent(params, device=self.config.device)


    def loader(self, fileNum):

        """Loads the label data"""

        filename = os.path.join(self.config.data_dir,
            self.config.inputName+str(fileNum).zfill(3) +
            '.npy')
        
        npData = np.flip(np.load(filename).swapaxes(1, 2), axis=2)
        torchData = torch.from_numpy(npData.astype('float32')).cuda(self.config.device)

        return torchData
    

    def converter(self):

        """Converts the data using squidtools and saves it"""

        # Iterating through the files specified
        for fileNum in range(self.config.numFiles):

            # Loading the file
            print("Converting File", fileNum)
            data = self.loader(fileNum)
            fileSize = data.shape[0]

            # Creating a progress bar
            imageProgress = tqdm.tqdm(total=fileSize,
                                      desc='Images', position=0)

            # Creating an ouput tensor with torch
            output = torch.from_numpy(np.ndarray([fileSize, self.config.dim,
                                 self.config.dim])).cuda(
                                     self.config.device)

            # Enumerating through the images
            for index, image in enumerate(data):
                # Breaking the image into x and y components
                Jx, Jy = image[:, :, 0], image[:, :, 1]

                # Simulating the magnetic image from the current density and
                # adding it to the output tensor
                signal = self.forward.forward(Jx, Jy)
                output[index] = signal

                imageProgress.update()

            # Reshaping and saving the output tensor
            output = output.reshape(fileSize,self.config.dim,
                                 self.config.dim, self.config.out_im_c)
            np.save(os.path.join(self.config.data_dir,
            self.config.outputName+str(fileNum).zfill(3) +
            '.npy'), output.detach().cpu().numpy())


if __name__ == "__main__":

    class config(object):
        data_dir = "/data/s4451856/training_data"
        inputName = "64x64_50um_Jxy_"
        outputName = "64x64_SQUID_B_"

        rho1 = 0.08 # 0.08 # 0.11
        rho2 = 0.26 # 0.26 # 0.515
        height = 0.35 # 0.35 # 0.225
        phi=np.deg2rad(62) # 62 # 66

        numFiles = 77
        dim = 64
        in_im_c = 3
        out_im_c = 1

        device = torch.device("cuda:1")

    converter(config).converter()
