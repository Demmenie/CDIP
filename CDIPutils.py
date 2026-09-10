import numpy as np
from sklearn.metrics import root_mean_squared_error as rmse
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import math
import torch


def SQUIDify(batch, angle, orient):

    """
    Changes the orientation of the image using the simple angle conversion.

    Input:
        - batch (np.array): The batch of images you want to convert.
        - angle (int): The angle you want to convert them to.
        - orient (int: 0 or 1): The component you want to include depending on
                                the orientation of the SQUID; 0 for x, 1 for y.
    
    Returns:
        - SQUIDbatch (torch.tensor): The converted batch of images.
    """

    # Finding the sin and cos of the SQUID angle
    sin = np.sin(np.deg2rad(angle))
    cos = np.cos(np.deg2rad(angle))

    # Converting the batch
    SQUIDBatch = torch.Tensor((sin * batch[:, orient, :, :]) +
                            (cos * batch[:, 2, :, :]))

    return torch.reshape(SQUIDBatch, (batch.shape[0], 1, batch.shape[2],
                                      batch.shape[3]))


def magnitude(image):

    """
    Converts a singular image with multiple components into a magnitude image.

    Input:
        - image (np.array/list): The image you want to convert.

    Output:
        - magImage (np.array): The magnitude of the input image.
    """

    imMag = []

    # Enumerating the rows of the images
    for rIndex, row in enumerate(image):
        imMag.append([])

        # Enumerating the pixels in the row
        for pIndex, pixel in enumerate(row):
            # Finding the magnitude of each pixel and adding it to the new row
            elMag = math.sqrt(sum(pow(element, 2) for element in pixel))
            imMag[rIndex].append(elMag)

    return np.array(imMag)


def batchMagnitude(batch):

    """
    Converts a whole batch of images into their magnitude.

    Input:
        - batch (np.array/list)

    Ouptut:
        - magBatch (np.array)
    """

    # Enumerating each image, calling the magnitude function and adding it to
    # new batch.
    batchMag = []
    for image in batch:
        magImage = magnitude(image)
        batchMag.append(magImage)

    return np.array(batchMag)


def accuracy(labelData, currentDensityPred, convMagnitude=True):

    """
    Measures the accuracy of a batch of predictions with SSIM, RMSE and PSNR.
    This only works on the magnitudes of the predictions/labels.

    Input:
        - labelData (np.array/list): The label images of the corresponding
            predictions.
        - currentDensityPred (np.array/list): The predicted current density
            images.
        - convMagnitude (bool): Indicator of if the data needs to be converted
            into magnitude or if it has been done already.

    Output:
        - avSSIM (float): The average of the SSIM of all the images.
        - avRMSE (float): The average of the RMSE of all the images.
        - avPSNR (float): The average of the PSNR of all the images.
    """

    ssimList = []
    rmseList = []
    psnrList = []

    # Enumerating the batch
    for index, CDPred in enumerate(currentDensityPred):

        # Finding the label image that corresponds to this prediction
        labelImage = labelData[index]

        # Converting the images to magnitude if necessary
        if convMagnitude == True:
            CDPred = magnitude(CDPred)
            labelImage = magnitude(labelImage)

        # Finding the SSIM, RMSE and PSNR of this prediction and adding them to
        # the list
        SSIM = ssim(labelImage, CDPred, data_range=CDPred.min() - CDPred.max())
        ssimList.append(SSIM)

        RMSE = rmse(labelImage, CDPred)
        rmseList.append(RMSE)

        PSNR = psnr(labelImage, CDPred, data_range=CDPred.min() - CDPred.max())
        psnrList.append(PSNR)

    # Finding the average of all the scores
    avSSIM = np.mean(ssimList)
    avRMSE = np.mean(rmseList)
    avPSNR = np.mean(psnrList)

    return avSSIM, avRMSE, avPSNR
