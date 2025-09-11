"""
Adaptive 3D U-Net for Hole-Filling in Galaxy Surveys
====================================================

This module defines an adaptive 3D U-Net model for inpainting missing data in 3D fields, such as those found in galaxy surveys. The model can take an additional mask channel as input to indicate missing regions and uses a ReLU activation function at the output to ensure non-negative predictions.
The model architecture is flexible, allowing for dynamic adjustment of depth based on input size and minimum feature map size. It also includes options for dropout regularization.
Author: [Edoardo Maragliano]
Date: [9 September 2025]

"""

import tensorflow as tf
from tensorflow import keras
import numpy as np


# =====================
# Adaptive 3D U-Net Model
# =====================
class InpaintingModel:
    """Adaptive 3D U-Net with mask input and ReLU output for non-negative predictions."""
    def __init__(self, base_filters=16, min_size=4, dropout_layer=False, dropout_rate=0.1, input_field='rho', norm_val=40):
        """
        Initializes the inpainting model.
        Parameters:
            base_filters (int): The number of base filters to use in the model's layers. Default is 16.
            min_size (int): The minimum spatial size for feature maps in the model. Default is 4.
            dropout_layer (bool): Whether to include a dropout layer in the model. Default is False.
            dropout_rate (float): The dropout rate to use if dropout_layer is True. Default is 0.1.
            input_field (str): Type of input field, either 'rho' for density or 'delta' for contrast. Default is 'rho'.
            norm_val (float): Normalization value used for shifted ReLU when input_field is 'delta'. Default is 40.
            
        Raises:
            ValueError: If input_field is not 'rho' or 'delta'.
        """

    
        self.base_filters = base_filters
        self.min_size = min_size
        self.dropout_layer = dropout_layer
        self.dropout_rate = dropout_rate
        self.norm_val = norm_val
    
        self.output_activation = self.shifted_relu if input_field == 'delta' else tf.nn.relu
        self.logger = None

    def shifted_relu(self, x):
        """Shifted ReLU activation to ensure outputs are >= -1/norm_val, i.e. delta >= -1.
        Use standard ReLU if you are training rho to rho.

        Parameters:
            x (tf.Tensor): Input tensor.
        Returns:
            tf.Tensor: Activated tensor.
        """
        min_val = -1.0/self.norm_val
        return tf.nn.relu(x - min_val) + min_val

    def set_logger(self, logger):
        self.logger = logger

    def prepare_model(self, input_size=(128,128,128,2)):
        """Prepares the Keras model for training.
        Parameters:
            input_size (tuple): The spatial dimensions of the input data. Default is (128, 128, 128, 2), i.e. (depth, height, width, channels).
        """

        inputs = keras.layers.Input(input_size)

        # Compute number of downsampling layers
        depth = int(np.floor(np.log2(min(input_size[:3]) / self.min_size)))
        if self.logger:
            self.logger.info(f"Model depth set to {depth} based on input size {input_size} and min_size {self.min_size}")

        # Create convolutional layers
        convs = []
        x = inputs
        filters = self.base_filters

        # Encoder path
        for d in range(depth):
            # Convolutional block
            x = keras.layers.Conv3D(filters, (3,3,3), activation='relu', padding='same')(x)
            x = keras.layers.Conv3D(filters, (3,3,3), activation='relu', padding='same')(x)
            convs.append(x)
            # Max pooling
            x = keras.layers.MaxPooling3D((2,2,2))(x)
            # Update filters
            filters *= 2

        # Bottleneck
        x = keras.layers.Conv3D(filters, (3,3,3), activation='relu', padding='same')(x)
        x = keras.layers.Conv3D(filters, (3,3,3), activation='relu', padding='same')(x)

        # Dropout
        if self.dropout_layer:
            x = keras.layers.Dropout(self.dropout_rate)(x)

        # Decoder path
        for d in reversed(range(depth)):
            # Reduce filters
            filters //= 2

            ## in Punya's code, it was (Convo then upsample) like in Paper 1
            # Convolutional block
            x = keras.layers.Conv3D(filters, (3,3,3), activation='relu', padding='same')(x)
            x = keras.layers.Conv3D(filters, (3,3,3), activation='relu', padding='same')(x)

            # Upsample
            x = keras.layers.Conv3DTranspose(filters, (3,3,3), strides=(2,2,2), padding='same')(x)
            x = keras.layers.concatenate([x, convs[d]], axis=-1)




        # Output layer (non-negative)
        outputs = keras.layers.Conv3D(1, (3,3,3), padding='same', activation=self.output_activation, dtype='float32')(x)

        return keras.models.Model(inputs=[inputs], outputs=[outputs])

# =====================
# Masked MSE Loss (optional)
# =====================
def masked_mse(y_true, y_pred, mask):
    missing = 1 - mask  # 1 where missing, 0 where observed
    return tf.reduce_mean(missing * tf.square(y_pred - y_true))
