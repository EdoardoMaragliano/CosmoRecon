"""
Adaptive 3D U-Net for Hole-Filling in Galaxy Surveys
====================================================

This module defines an adaptive 3D U-Net model for inpainting missing data in 3D fields, such as those found in galaxy surveys. 
The model can handle inputs with missing regions specified by a binary mask and is designed to produce non-negative outputs using ReLU activations.
The model architecture is flexible, allowing for dynamic adjustment of depth based on input size and minimum feature map size. 
It also includes options for dropout regularization.
Author: [Edoardo Maragliano]
Date: [9 September 2025]

"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import logging

logger = logging.getLogger(__name__)

from CosmoFiller.utils.loggers import setup_logger

# -------------------------
# Module logger
# -------------------------
logger = setup_logger(__name__)

# =====================
# Adaptive 3D U-Net Model
# =====================
class UNet:
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

####################################################
# MASKED INPAINTING MODEL (BASED ON UNet MODEL)
####################################################

# =============================
# MaskedInpaintingUNet
# =============================
class MaskedInpaintingUNet:
    """
    MaskedInpaintingUNet
    --------------------

    3D U-Net for inpainting with optional mask support. This Keras Model
    subclass combines the U-Net architecture and the masked-training logic
    so the same object can be used for both forward inference and training
    with or without a binary/float mask specifying observed (1) vs missing (0)
    voxels.

    Key behavior
    - When instantiated with use_mask=True, train_step and test_step expect
      batches of the form (x, y_true, mask). When use_mask=False they expect
      (x, y_true).
    - The model builds an internal Keras Model implementing the U-Net and
      delegates the forward pass to it.
    - A shifted ReLU output activation is used when working with delta fields
      (to enforce the lower bound -1/norm_val); otherwise a standard ReLU is used.
    - The masked MSE loss computes mean squared error only over missing voxels
      (mask == 0), avoiding contamination from observed regions.

    Parameters
    ----------
    input_size : tuple
        Spatial input shape including channels, e.g. (D, H, W, C).
    base_filters : int
        Number of filters in the first conv block, doubled at each downsampling.
    min_size : int
        Minimum spatial size used to determine U-Net depth.
    dropout_layer : bool
        If True, apply dropout in the bottleneck.
    dropout_rate : float
        Dropout rate applied when dropout_layer is True.
    input_field : {'rho', 'delta'}
        If 'delta', use shifted ReLU output to enforce delta >= -1/norm_val.
    norm_val : float
        Normalization constant used to compute the shifted ReLU offset.
    use_mask : bool
        If True, training/test steps expect and use a mask tensor.
    logger : logger.Logger or None
        Optional logger for informational messages.

    

    """
    def __init__(self, 
                input_size=(128,128,128,1),
                base_filters=16, min_size=4,
                dropout_layer=False, dropout_rate=0.1,
                input_field='rho', 
                norm_val=40, 
                use_mask=False, 
                logger=None,
                ):
        super().__init__()
        self.use_mask = use_mask
        self.logger = logger
        self.norm_val = norm_val


        # output activation
        self.output_activation = self.shifted_relu if input_field=='delta' else tf.nn.relu

        # costruzione della U-Net interna
        self.unet = self._build_unet(input_size=input_size,
                                     base_filters=base_filters,
                                     min_size=min_size,
                                     dropout_layer=dropout_layer,
                                     dropout_rate=dropout_rate)
        

    # ----------------------------
    # shifted ReLU per delta
    # ----------------------------
    def shifted_relu(self, x):
        min_val = -1.0/self.norm_val
        return tf.nn.relu(x - min_val) + min_val


    # ----------------------------
    # Costruzione U-Net 3D
    # ----------------------------
    def _build_unet(self, input_size, base_filters=16, min_size=4,
                    dropout_layer=False, dropout_rate=0.1):
        inputs = keras.layers.Input(shape=input_size)
        depth = int(np.floor(np.log2(min(input_size[:3]) / min_size)))
        if self.logger:
            self.logger.info(f"U-Net depth: {depth}")

        convs = []
        x = inputs
        filters = base_filters

        # Encoder
        for d in range(depth):
            x = keras.layers.Conv3D(filters, kernel_size=(3,3,3), activation='relu', padding='same')(x)
            x = keras.layers.Conv3D(filters, kernel_size=(3,3,3), activation='relu', padding='same')(x)
            convs.append(x)
            x = keras.layers.MaxPooling3D((2,2,2))(x)
            filters *= 2

        # Bottleneck
        x = keras.layers.Conv3D(filters, (3,3,3), activation='relu', padding='same')(x)
        x = keras.layers.Conv3D(filters, (3,3,3), activation='relu', padding='same')(x)
        if dropout_layer:
            x = keras.layers.Dropout(dropout_rate)(x)

        # Decoder
        for d in reversed(range(depth)):
            filters //= 2
            x = keras.layers.Conv3D(filters, (3,3,3), activation='relu', padding='same')(x)
            x = keras.layers.Conv3D(filters, (3,3,3), activation='relu', padding='same')(x)
            x = keras.layers.Conv3DTranspose(filters, (3,3,3), strides=(2,2,2), padding='same')(x)
            x = keras.layers.concatenate([x, convs[d]], axis=-1)

        # Output
        outputs = keras.layers.Conv3D(1, (3,3,3), padding='same', activation=self.output_activation, dtype='float32')(x)

        return keras.Model(inputs=inputs, outputs=outputs)


# ============================

class MaskedMSE(tf.keras.losses.Loss):
    """
    MaskedMSE for single mask, either in the 1 or two channel configuration
    Computes Mean Squared Error only over missing voxels (mask == 0).
    """

    def __init__(self, mask, **kwargs):
        super().__init__(**kwargs)
        mask = tf.cast(mask, tf.float32)   # (D,H,W,1)
        self.mask = tf.Variable(mask, trainable=False, dtype=tf.float32, name='mse_mask')

    def call(self, y_true, y_pred):
        diff = tf.square(y_true - y_pred)
        masked_diff = diff * (1 - self.mask)
        denom = tf.reduce_sum(1 - self.mask) + 1e-8
        return tf.reduce_sum(masked_diff) / denom
    
    def get_config(self):
        # Keras non può serializzare direttamente il tensore della mask
        config = super().get_config()
        config.update({"mask": None})  # serve solo per compatibilità
        return config
    
    
def compute_gradient(x):
    # assume x: [batch, H, W, D, channels]
    
    gx = x[:, 1:, :, :, :] - x[:, :-1, :, :, :]     # grad along x, shape [batch, H-1, W, D, channels]
    gy = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
    gz = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
    

    return gx, gy, gz


def dilate_mask(mask, iterations=1):
    """
    Dilata la maschera usando MaxPool3D.
    mask: [batch, H, W, D] o [batch, H, W, D, 1]
    """
    logger.info(f"Dilating mask with {iterations} iterations.")
    # Assicuriamo che ci sia il canale alla fine
    if len(mask.shape) == 4:
        mask = tf.expand_dims(mask, axis=-1)
    
    # Kernel 3x3x3 equivale a guardare i vicini immediati
    # Padding SAME mantiene le dimensioni originali
    for _ in range(iterations):
        mask = tf.nn.max_pool3d(
            mask, 
            ksize=[1, 3, 3, 3, 1], 
            strides=[1, 1, 1, 1, 1], 
            padding='SAME'
        )
        
    return tf.squeeze(mask, axis=-1)

def prepare_mask_tensor(mask_array):
    """
    Utility per convertire la maschera numpy in tensore broadcastabile (1, H, W, D, 1).
    """
    mask_tensor = tf.cast(mask_array, tf.float32)
    
    # Gestione dimensioni per garantire (1, H, W, D, 1)
    if len(mask_tensor.shape) == 3: # (H, W, D)
        mask_tensor = mask_tensor[None, ..., None]
    elif len(mask_tensor.shape) == 4: # (H, W, D, 1) o (1, H, W, D)
        if mask_tensor.shape[0] != 1: 
             mask_tensor = mask_tensor[None, ...] # Aggiungi batch dim se manca
        if mask_tensor.shape[-1] != 1:
             mask_tensor = mask_tensor[..., None] # Aggiungi channel dim se manca
             
    return mask_tensor


class MaskedGradientLoss(tf.keras.losses.Loss):
    def __init__(self, mask, dilation_iter=2, **kwargs):
        """
        mask: np.array (0=Buco, 1=Valido)
        dilation_iter: Quanti pixel espandere la zona di controllo attorno al buco
        """
        super().__init__(**kwargs)
        self.dilation_iter = dilation_iter
        
        # 1. Preparazione e Inversione (Hole = 1)
        mask_tensor = prepare_mask_tensor(mask)
        hole_mask = 1.0 - mask_tensor
        
        # 2. Dilatazione "One-Off" (fatta solo ora, non durante il training)
        dilated_hole = hole_mask
        for _ in range(dilation_iter):
            dilated_hole = tf.nn.max_pool3d(
                dilated_hole, 
                ksize=[1, 3, 3, 3, 1], 
                strides=[1, 1, 1, 1, 1], 
                padding='SAME'
            )
        
        # 3. Slicing per allineamento con i gradienti
        # Rimuoviamo la batch dim (1) temporaneamente per slicing più pulito
        dh = tf.squeeze(dilated_hole, axis=0) 
        
        # FIX: Wrap these in tf.Variable(trainable=False)
        # This forces them to be registered as part of the model state on the GPU
        self.mask_x = tf.convert_to_tensor(dh[1:, :, :, :][None, ...], dtype=tf.float32, name='grad_mask_x')
        self.mask_y = tf.convert_to_tensor(dh[:, 1:, :, :][None, ...], dtype=tf.float32, name='grad_mask_y')
        self.mask_z = tf.convert_to_tensor(dh[:, :, 1:, :][None, ...], dtype=tf.float32, name='grad_mask_z')

            
        # 4. Pre-calcolo denominatori (Scalar variables)
        self.denom_x = tf.constant(tf.reduce_sum(self.mask_x) + 1e-8, dtype=tf.float32)
        self.denom_y = tf.constant(tf.reduce_sum(self.mask_y) + 1e-8, dtype=tf.float32)
        self.denom_z = tf.constant(tf.reduce_sum(self.mask_z) + 1e-8, dtype=tf.float32)


    def call(self, y_true, y_pred):
        # Calcolo gradienti
        gx_t, gy_t, gz_t = compute_gradient(y_true)
        gx_p, gy_p, gz_p = compute_gradient(y_pred)

        # Broadcasting works automatically here between (Batch, H, W, D) and (1, H, W, D)
        term_x = tf.reduce_sum(tf.square(gx_t - gx_p) * self.mask_x) / self.denom_x
        term_y = tf.reduce_sum(tf.square(gy_t - gy_p) * self.mask_y) / self.denom_y
        term_z = tf.reduce_sum(tf.square(gz_t - gz_p) * self.mask_z) / self.denom_z

        return term_x + term_y + term_z
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "mask": None,
            "dilation_iter": self.dilation_iter
        })
        return config



class MaskedMSEWithGradient(tf.keras.losses.Loss):
    def __init__(self, mask, gradient_weight=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mse_loss_fn = MaskedMSE(mask)
        self.grad_loss_fn = MaskedGradientLoss(mask)
        self.gradient_weight = gradient_weight

    def call(self, y_true, y_pred):
        mse_loss = self.mse_loss_fn(y_true, y_pred)
        grad_loss = self.grad_loss_fn(y_true, y_pred)
        return mse_loss + self.gradient_weight * grad_loss
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "gradient_weight": self.gradient_weight
        })
        return config
