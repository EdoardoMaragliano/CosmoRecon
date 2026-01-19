"""
Adaptive 2D U-Net for Hole-Filling in Galaxy Surveys (2D Slice Version)
=======================================================================

This module defines an adaptive 2D U-Net model for inpainting missing data in 2D slices.
It mirrors the functionality of the 3D module but uses 2D convolutions and pooling.
Ideal for debugging, testing on slices, or processing projected maps.

Author: [Edoardo Maragliano]
Date: [14 January 2026]
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import logging

from CosmoFiller.utils.loggers import setup_logger

# -------------------------
# Module logger
# -------------------------
logger = setup_logger(__name__)

# =====================
# Adaptive 2D U-Net Model
# =====================
class UNet2D:
    """Adaptive 2D U-Net with mask input and ReLU output for non-negative predictions."""
    def __init__(self, base_filters=16, min_size=4, dropout_layer=False, dropout_rate=0.1, input_field='rho', norm_val=40):
        """
        Initializes the 2D inpainting model.
        Parameters match the 3D version, but applied to 2D inputs.
        """
        self.base_filters = base_filters
        self.min_size = min_size
        self.dropout_layer = dropout_layer
        self.dropout_rate = dropout_rate
        self.norm_val = norm_val
    
        self.output_activation = self.shifted_relu if input_field == 'delta' else tf.nn.relu
        self.logger = None

    def shifted_relu(self, x):
        """Shifted ReLU activation."""
        min_val = -1.0/self.norm_val
        return tf.nn.relu(x - min_val) + min_val

    def set_logger(self, logger):
        self.logger = logger

    def prepare_model(self, input_size=(128,128,2)):
        """Prepares the Keras model for training (2D version).
        
        Parameters:
            input_size (tuple): (height, width, channels). Default (128, 128, 2).
        """

        inputs = keras.layers.Input(input_size)

        # Compute depth based on H/W
        depth = int(np.floor(np.log2(min(input_size[:2]) / self.min_size)))
        if self.logger:
            self.logger.info(f"Model 2D depth set to {depth} based on input size {input_size} and min_size {self.min_size}")

        convs = []
        x = inputs
        filters = self.base_filters

        # Encoder path
        for d in range(depth):
            x = keras.layers.Conv2D(filters, (3,3), activation='relu', padding='same')(x)
            x = keras.layers.Conv2D(filters, (3,3), activation='relu', padding='same')(x)
            convs.append(x)
            x = keras.layers.MaxPooling2D((2,2))(x)
            filters *= 2

        # Bottleneck
        x = keras.layers.Conv2D(filters, (3,3), activation='relu', padding='same')(x)
        x = keras.layers.Conv2D(filters, (3,3), activation='relu', padding='same')(x)

        if self.dropout_layer:
            x = keras.layers.Dropout(self.dropout_rate)(x)

        # Decoder path
        for d in reversed(range(depth)):
            filters //= 2
            x = keras.layers.Conv2D(filters, (3,3), activation='relu', padding='same')(x)
            x = keras.layers.Conv2D(filters, (3,3), activation='relu', padding='same')(x)
            
            x = keras.layers.Conv2DTranspose(filters, (3,3), strides=(2,2), padding='same')(x)
            x = keras.layers.concatenate([x, convs[d]], axis=-1)

        # Output layer
        outputs = keras.layers.Conv2D(1, (3,3), padding='same', activation=self.output_activation, dtype='float32')(x)

        return keras.models.Model(inputs=[inputs], outputs=[outputs])


# =============================
# MaskedInpaintingUNet 2D
# =============================
class MaskedInpaintingUNet2D:
    """
    MaskedInpaintingUNet2D
    ----------------------
    2D Version of the Inpainting U-Net.
    """
    def __init__(self, 
                input_size=(128,128,1),
                base_filters=16, min_size=4,
                dropout_layer=False, dropout_rate=0.1,
                input_field='rho', 
                norm_val=40, 
                use_mask=False, 
                logger=None,
                ):
        self.use_mask = use_mask
        self.logger = logger
        self.norm_val = norm_val

        self.output_activation = self.shifted_relu if input_field=='delta' else tf.nn.relu

        self.unet = self._build_unet(input_size=input_size,
                                     base_filters=base_filters,
                                     min_size=min_size,
                                     dropout_layer=dropout_layer,
                                     dropout_rate=dropout_rate)
        
    def shifted_relu(self, x):
        min_val = -1.0/self.norm_val
        return tf.nn.relu(x - min_val) + min_val

    def _build_unet(self, input_size, base_filters=16, min_size=4,
                    dropout_layer=False, dropout_rate=0.1):
        
        inputs = keras.layers.Input(shape=input_size)
        
        # Calculate depth (using first 2 dims only)
        depth = int(np.floor(np.log2(min(input_size[:2]) / min_size)))
        
        if self.logger:
            self.logger.info(f"U-Net 2D depth: {depth}")

        convs = []
        x = inputs
        filters = base_filters

        # Encoder
        for d in range(depth):
            x = keras.layers.Conv2D(filters, kernel_size=(3,3), activation='relu', padding='same')(x)
            x = keras.layers.Conv2D(filters, kernel_size=(3,3), activation='relu', padding='same')(x)
            convs.append(x)
            x = keras.layers.MaxPooling2D((2,2))(x)
            filters *= 2

        # Bottleneck
        x = keras.layers.Conv2D(filters, (3,3), activation='relu', padding='same')(x)
        x = keras.layers.Conv2D(filters, (3,3), activation='relu', padding='same')(x)
        if dropout_layer:
            x = keras.layers.Dropout(dropout_rate)(x)

        # Decoder
        for d in reversed(range(depth)):
            filters //= 2
            x = keras.layers.Conv2D(filters, (3,3), activation='relu', padding='same')(x)
            x = keras.layers.Conv2D(filters, (3,3), activation='relu', padding='same')(x)
            x = keras.layers.Conv2DTranspose(filters, (3,3), strides=(2,2), padding='same')(x)
            x = keras.layers.concatenate([x, convs[d]], axis=-1)

        # Output
        outputs = keras.layers.Conv2D(1, (3,3), padding='same', activation=self.output_activation, dtype='float32')(x)

        return keras.Model(inputs=inputs, outputs=outputs)


# ============================
# 2D Loss Utilities
# ============================

class MaskedMSE(tf.keras.losses.Loss):
    """
    MaskedMSE works for 2D as well since element-wise ops are rank-agnostic.
    Re-implemented here for completeness.
    """
    def __init__(self, mask, **kwargs):
        super().__init__(**kwargs)
        mask = tf.cast(mask, tf.float32)
        # Ensure mask is variable (not re-initialized every call)
        self.mask = tf.Variable(mask, trainable=False, dtype=tf.float32, name='mse_mask_2d')

    def call(self, y_true, y_pred):
        diff = tf.square(y_true - y_pred)
        masked_diff = diff * (1 - self.mask)
        denom = tf.reduce_sum(1 - self.mask) + 1e-8
        return tf.reduce_sum(masked_diff) / denom
    
    def get_config(self):
        config = super().get_config()
        config.update({"mask": None})
        return config

def compute_gradient_2d(x):
    """
    Computes gradients only along H and W (2D).
    Assume x: [batch, H, W, channels]
    """
    gx = x[:, 1:, :, :] - x[:, :-1, :, :]  # grad along H
    gy = x[:, :, 1:, :] - x[:, :, :-1, :]  # grad along W
    return gx, gy

def dilate_mask_2d(mask, iterations=1):
    """
    Dilates a 2D mask using MaxPool2D.
    mask: [batch, H, W] or [batch, H, W, 1]
    """
    logger.info(f"Dilating 2D mask with {iterations} iterations.")
    if len(mask.shape) == 3:
        mask = tf.expand_dims(mask, axis=-1)
    
    for _ in range(iterations):
        mask = tf.nn.max_pool2d(
            mask, 
            ksize=[1, 3, 3, 1], 
            strides=[1, 1, 1, 1], 
            padding='SAME'
        )
    return tf.squeeze(mask, axis=-1)

def prepare_mask_tensor_2d(mask_array):
    """
    Convert numpy mask to (1, H, W, 1) tensor.
    """
    mask_tensor = tf.cast(mask_array, tf.float32)
    
    # Handle dimensions: (H,W) -> (1,H,W,1)
    if len(mask_tensor.shape) == 2: # (H, W)
        mask_tensor = mask_tensor[None, ..., None]
    elif len(mask_tensor.shape) == 3: 
        # Case (H, W, 1) -> add batch
        if mask_tensor.shape[-1] == 1:
            mask_tensor = mask_tensor[None, ...]
        # Case (Batch, H, W) -> add channel
        else:
             mask_tensor = mask_tensor[..., None]
             
    return mask_tensor

class MaskedGradientLoss2D(tf.keras.losses.Loss):
    def __init__(self, mask, dilation_iter=2, **kwargs):
        """
        2D Gradient Loss: Only penalizes grad_x and grad_y.
        """
        super().__init__(**kwargs)
        self.dilation_iter = dilation_iter
        
        # 1. Prepare Mask
        mask_tensor = prepare_mask_tensor_2d(mask)
        hole_mask = 1.0 - mask_tensor
        
        # 2. Dilation
        dilated_hole = hole_mask
        for _ in range(dilation_iter):
            dilated_hole = tf.nn.max_pool2d(
                dilated_hole, 
                ksize=[1, 3, 3, 1], 
                strides=[1, 1, 1, 1], 
                padding='SAME'
            )
        
        # 3. Slicing
        # Remove batch dim for slicing, assuming mask is constant across batch
        dh = tf.squeeze(dilated_hole, axis=0) # (H, W, 1)
        
        # Create masks for gradients (shifted by 1 pixel)
        # Note: tf.Variable used to track state properly
        self.mask_x = tf.convert_to_tensor(dh[1:, :, :][None, ...], dtype=tf.float32, name='grad_mask_x_2d')
        self.mask_y = tf.convert_to_tensor(dh[:, 1:, :][None, ...], dtype=tf.float32, name='grad_mask_y_2d')
            
        # 4. Denominators
        self.denom_x = tf.constant(tf.reduce_sum(self.mask_x) + 1e-8, dtype=tf.float32)
        self.denom_y = tf.constant(tf.reduce_sum(self.mask_y) + 1e-8, dtype=tf.float32)

    def call(self, y_true, y_pred):
        gx_t, gy_t = compute_gradient_2d(y_true)
        gx_p, gy_p = compute_gradient_2d(y_pred)

        term_x = tf.reduce_sum(tf.square(gx_t - gx_p) * self.mask_x) / self.denom_x
        term_y = tf.reduce_sum(tf.square(gy_t - gy_p) * self.mask_y) / self.denom_y

        return term_x + term_y
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "mask": None,
            "dilation_iter": self.dilation_iter
        })
        return config

class MaskedMSEWithGradient2D(tf.keras.losses.Loss):
    def __init__(self, mask, gradient_weight=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mse_loss_fn = MaskedMSE(mask)
        self.grad_loss_fn = MaskedGradientLoss2D(mask)
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