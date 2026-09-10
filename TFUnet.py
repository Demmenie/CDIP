import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import *
import copy


def TFUnet(config):

    """
    This function creates a UNET model based on the design from MAGIC_UNET, in 
    such a way that it is optimisable through the config. This function is no
    longer used for CDIP and was only meant to replicate MAGIC UNet.

    Input:
    - config (Object)

    Output:
    - model (Object)
    """

    with tf.device('/GPU:0'):
        keras.utils.set_random_seed(config.seed)

        # A layer that can take the image size as input
        inputLayer = keras.Input(shape=(config.im_h,config.im_w,config.in_im_c), name='input')

        # Encoding/Downsampling layers
        prevLayer = inputLayer
        downLayers = []
        for index, layer in enumerate(config.layers):
            # print("Encoding", index, layer, config.pooling[index], config.dropouts[index])

            conv = Conv2D(layer, 3, activation='tanh',
                padding='same', kernel_initializer='he_normal')(prevLayer)

            conv = Conv2D(layer, 3, activation = 'tanh',
                padding='same', kernel_initializer='he_normal')(conv)
            
            # All layers have a pooling layer and some have dropout. We can set this
            # as a hyperparameter
            if config.dropouts[index] != 0 and config.pooling[index]:
                drop = Dropout(config.dropouts[index])(conv)
                pool = AveragePooling2D(pool_size=(2, 2))(drop)

                # Prepending the layers so that they are reversed for decoding
                downLayers.insert(0, drop)
                prevLayer = pool
            
            elif config.dropouts[index] != 0 and not config.pooling[index]:
                drop = Dropout(config.dropouts[index])(conv)

                downLayers.insert(0, drop)
                prevLayer = drop

            else:
                pool = AveragePooling2D(pool_size=(2, 2))(conv)

                downLayers.insert(0, conv)
                prevLayer = pool

        # Creating a middle feature space layer, which has the same size as the
        # last encoding layer.
        conv = Conv2D(config.layers[-1], 3, activation='tanh',
                padding='same', kernel_initializer='he_normal')(prevLayer)
        conv = Conv2D(config.layers[-1], 3, activation='tanh',
                padding='same', kernel_initializer='he_normal')(conv)
        drop = Dropout(config.dropouts[-1])(conv)
        prevLayer = drop

        # Decoding/Upsampling layers
        for index, layer in enumerate(reversed(config.layers)):
            # print("Decoding", index, layer)

            up = Conv2D(layer, 2, activation='tanh', padding='same',
                kernel_initializer='he_normal')(UpSampling2D(size = (2,2))(prevLayer))
            
            # This merge creates the skip connections
            merge = concatenate([downLayers[index], up], axis=3)

            conv = Conv2D(layer, 3, activation='tanh', padding='same',
                kernel_initializer='he_normal')(merge)
            conv = Conv2D(layer, 3, activation='tanh', padding='same',
                kernel_initializer='he_normal')(conv)

            prevLayer = conv


        # Creating an output layer
        conv = Conv2D(config.out_im_c, 3, activation='tanh', padding='same',
            kernel_initializer='he_normal')(prevLayer)
        outputLayer = Conv2D(config.out_im_c, 1, activation='tanh')(conv)

        model = keras.Model(inputs=inputLayer, outputs=outputLayer, name='output')
        model.summary()
        return model


if __name__ == "__main__":
    class config:
        im_h = 64
        im_w = 64
        in_im_c = 3
        out_im_c = 2

        layers = [64, 64, 64, 64]
        dropouts = [0, 0, 0, 0.5, 0.5]
        pooling = [True, True, True, True]

    CDIPmodel = model(config)
    CDIPmodel.save()