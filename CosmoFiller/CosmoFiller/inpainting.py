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
import logging

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

# -----------------------------
# Masked MSE loss used by model
# -----------------------------
def masked_mse_loss(y_true, y_pred, mask):
    """Masked MSE: average only on missing voxels (mask==0)."""
    # mask: 1 where observed, 0 where missing
    missing = 1.0 - mask
    se = tf.square(y_true - y_pred) * missing
    denom = tf.reduce_sum(missing) + 1e-8
    return tf.reduce_sum(se) / denom


# -----------------------------
# Custom Model to support masked training
# -----------------------------
class MaskedInpaintingModel(keras.Model):
    """
    MaskedInpaintingModel(keras.Model)

    A Keras Model subclass that implements custom training and testing steps to
    support optional masked inpainting losses.

    This model expects its forward pass (call) to produce predictions compatible
    with the target y tensors. It provides an option to compute a masked mean
    squared error when a binary/float mask is supplied.

    Parameters
    ----------
    use_mask : bool, optional
        If True, the model's train_step and test_step expect incoming batch data
        to be tuples of (x, y_true, mask). If False, they expect (x, y_true).
        Defaults to False.
    *args, **kwargs :
        Additional positional and keyword arguments forwarded to keras.Model.

    Behavior
    --------
    - train_step(data):
        - If use_mask is True, unpacks data as (x, y_true, mask). Otherwise uses
          (x, y_true) and sets mask = None.
        - Runs a forward pass y_pred = self(x, training=True).
        - Computes loss:
            - If use_mask: calls masked_mse_loss(y_true, y_pred, mask).
            - Else: computes standard mean squared error: mean(square(y_true - y_pred)).
        - Adds any model regularization losses present in self.losses.
        - Computes gradients and applies them via self.optimizer.
        - Updates compiled metrics via self.compiled_metrics.update_state(y_true, y_pred).
        - Returns a dict mapping metric names to scalar results and includes a
          'loss' entry holding the computed loss tensor.

    - test_step(data):
        - Same unpacking logic as train_step.
        - Runs forward pass with training=False and computes loss using the same
          masked or unmasked MSE logic.
        - Updates compiled metrics and returns a dict of metric results including 'loss'.

    Example usage
    -------------
    - When not using masks:
        model = MaskedInpaintingModel(..., use_mask=False)
        model.compile(optimizer=..., metrics=[...])
        model.fit(dataset_of_pairs)  # dataset yields (x, y)

    - When using masks:
        model = MaskedInpaintingModel(..., use_mask=True)
        model.compile(optimizer=..., metrics=[...])
        model.fit(dataset_of_triplets)  # dataset yields (x, y, mask)

    - In combo with UNet:
        model_obj = UNet(
            base_filters=args.base_filters,
            min_size=args.min_size,
            dropout_layer=args.dropout,
            dropout_rate=args.dropout_rate,
            input_field=args.input_field,
            norm_val=args.density_normalization
        )
        model_obj.set_logger(logging)
        base_unet = model_obj.prepare_model(input_size=input_size)

        # Wrap in custom model for masked training
        masked_model = MaskedInpaintingModel(
            inputs=base_unet.input, 
            outputs=base_unet.output, 
            use_mask=use_mask)

        optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)

        masked_model.compile(
            optimizer=optimizer, 
            metrics=[keras.metrics.MeanSquaredError(name='mse')])


    """

    def __init__(self, *args, use_mask=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_mask = use_mask

    def train_step(self, data):
        # data can be (x, y) or (x, y, mask)
        if self.use_mask:
            x, y_true, mask = data
        else:
            x, y_true = data
            mask = None

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            if self.use_mask:
                loss = masked_mse_loss(y_true, y_pred, mask)
            else:
                loss = tf.reduce_mean(tf.square(y_true - y_pred))

            # add regularization losses if present
            loss += sum(self.losses)

        trainable_vars = self.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        # metrics
        self.compiled_metrics.update_state(y_true, y_pred)
        metrics = {m.name: m.result() for m in self.metrics}
        metrics['loss'] = loss
        return metrics

    def test_step(self, data):
        if self.use_mask:
            x, y_true, mask = data
            y_pred = self(x, training=False)
            loss = masked_mse_loss(y_true, y_pred, mask)
        else:
            x, y_true = data
            y_pred = self(x, training=False)
            loss = tf.reduce_mean(tf.square(y_true - y_pred))

        self.compiled_metrics.update_state(y_true, y_pred)
        metrics = {m.name: m.result() for m in self.metrics}
        metrics['loss'] = loss
        return metrics



# =============================
# MaskedInpaintingUNet
# =============================
class MaskedInpaintingUNet(keras.Model):
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
    logger : logging.Logger or None
        Optional logger for informational messages.

    Public methods
    --------------
    call(inputs, training=False)
        Forward pass delegating to the internal U-Net model.
    train_step(data)
        Custom train step supporting optional masks.
    test_step(data)
        Custom test step supporting optional masks.
    masked_mse_loss(y_true, y_pred, mask)
        Compute MSE averaged only over missing voxels (mask == 0).

    """
    def __init__(self, input_size=(128,128,128,2), base_filters=16, min_size=4,
                 dropout_layer=False, dropout_rate=0.1,
                 input_field='rho', norm_val=40, use_mask=False, logger=None):
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
            x = keras.layers.Conv3D(filters, (3,3,3), activation='relu', padding='same')(x)
            x = keras.layers.Conv3D(filters, (3,3,3), activation='relu', padding='same')(x)
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

    # ----------------------------
    # Forward pass
    # ----------------------------
    def call(self, inputs, training=False):
        return self.unet(inputs, training=training)

    # ----------------------------
    # Custom train step
    # ----------------------------
    def train_step(self, data):
        if self.use_mask:
            x, y_true, mask = data
        else:
            x, y_true = data
            mask = None

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            if self.use_mask:
                loss = self.masked_mse_loss(y_true, y_pred, mask)
            else:
                loss = tf.reduce_mean(tf.square(y_true - y_pred))
            loss += sum(self.losses)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # aggiornamento metriche
        self.compiled_metrics.update_state(y_true, y_pred)
        metrics = {m.name: m.result() for m in self.metrics}
        metrics['loss'] = loss
        return metrics

    # ----------------------------
    # Custom test step
    # ----------------------------
    def test_step(self, data):
        if self.use_mask:
            x, y_true, mask = data
            y_pred = self(x, training=False)
            loss = self.masked_mse_loss(y_true, y_pred, mask)
        else:
            x, y_true = data
            y_pred = self(x, training=False)
            loss = tf.reduce_mean(tf.square(y_true - y_pred))

        self.compiled_metrics.update_state(y_true, y_pred)
        metrics = {m.name: m.result() for m in self.metrics}
        metrics['loss'] = loss
        return metrics

    # ----------------------------
    # Masked MSE Loss
    # ----------------------------
    def masked_mse_loss(self, y_true, y_pred, mask):
        missing = 1.0 - mask
        squared_error = tf.square(y_true - y_pred)
        masked_error = squared_error * missing
        return tf.reduce_sum(masked_error) / (tf.reduce_sum(missing) + 1e-8)
