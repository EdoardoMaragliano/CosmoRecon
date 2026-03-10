"""
Adaptive 3D U-Net for Cosmological Field Reconstruction
========================================================

Defines U-Net architectures and custom loss functions for reconstructing
3D cosmological density fields (e.g. RSD removal, field inpainting).

Two model classes are provided:
  - ``UNet``: standard 3D U-Net with configurable depth and output activation.
  - ``MaskedUNet3D``: wrapper that delegates to ``UNet`` for model
    construction and pairs it with mask-aware training logic.

Custom losses:
  - ``MaskedMSE``: MSE restricted to masked voxels (mask == 0).
  - ``MaskedGradientLoss``: gradient-matching loss near mask boundaries.
  - ``MaskedMSEWithGradient``: weighted combination of the two above.

Author: Edoardo Maragliano
Date:   September 2025
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import Callable, Dict, Optional, Tuple, Union

from CosmoRecon.utils.loggers import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def compute_depth(spatial_dims: Tuple[int, ...], min_size: int) -> int:
    """Compute U-Net depth from spatial dimensions and minimum feature size.

    Parameters
    ----------
    spatial_dims : tuple of int
        The spatial dimensions (H, W, D) of the input.
    min_size : int
        Minimum feature-map size at the bottleneck.

    Returns
    -------
    int
        Number of encoder/decoder levels.
    """
    return int(np.floor(np.log2(min(spatial_dims) / min_size)))


def compute_gradient(x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Compute finite-difference spatial gradients along each axis.

    Parameters
    ----------
    x : tf.Tensor
        Tensor of shape ``(batch, H, W, D, channels)``.

    Returns
    -------
    gx, gy, gz : tf.Tensor
        Gradient tensors with one fewer element along the differentiated axis.
    """
    gx = x[:, 1:, :, :, :] - x[:, :-1, :, :, :]
    gy = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
    gz = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
    return gx, gy, gz


def dilate_mask(mask: tf.Tensor, iterations: int = 1) -> tf.Tensor:
    """Dilate a binary mask using 3D max-pooling.

    Parameters
    ----------
    mask : tf.Tensor
        Shape ``(batch, H, W, D)`` or ``(batch, H, W, D, 1)``.
    iterations : int
        Number of dilation passes.

    Returns
    -------
    tf.Tensor
        Dilated mask with shape ``(batch, H, W, D)``.
    """
    logger.debug("Dilating mask with %d iterations.", iterations)
    if len(mask.shape) == 4:
        mask = tf.expand_dims(mask, axis=-1)
    for _ in range(iterations):
        mask = tf.nn.max_pool3d(
            mask,
            ksize=[1, 3, 3, 3, 1],
            strides=[1, 1, 1, 1, 1],
            padding='SAME',
        )
    return tf.squeeze(mask, axis=-1)


def prepare_mask_tensor(mask_array: Union[np.ndarray, tf.Tensor]) -> tf.Tensor:
    """Convert a numpy mask to a broadcastable tensor of shape ``(1, H, W, D, 1)``.

    Parameters
    ----------
    mask_array : array-like
        Mask with 3 to 5 dimensions.

    Returns
    -------
    tf.Tensor
        Float32 tensor of shape ``(1, H, W, D, 1)``.
    """
    mask_tensor = tf.cast(mask_array, tf.float32)
    if len(mask_tensor.shape) == 3:
        mask_tensor = mask_tensor[None, ..., None]
    elif len(mask_tensor.shape) == 4:
        if mask_tensor.shape[0] != 1:
            mask_tensor = mask_tensor[None, ...]
        if mask_tensor.shape[-1] != 1:
            mask_tensor = mask_tensor[..., None]
    return mask_tensor


def shifted_relu(norm_val: float) -> Callable[[tf.Tensor], tf.Tensor]:
    """Return a shifted-ReLU activation enforcing output >= -1/norm_val.

    This is used for delta-field reconstruction where the physical constraint
    is delta >= -1 (i.e. normalised_delta >= -1/norm_val).

    Parameters
    ----------
    norm_val : float
        Normalisation constant.

    Returns
    -------
    callable
        Activation function.
    """
    min_val = -1.0 / norm_val

    def activation(x):
        return tf.nn.relu(x - min_val) + min_val

    return activation


def build_unet(input_size: Tuple[int, ...], base_filters: int = 16, min_size: int = 4,
               dropout_layer: bool = False, dropout_rate: float = 0.1,
               output_activation: Callable = tf.nn.relu,
               logger_obj: Optional[object] = None) -> keras.Model:
    """Build a 3D U-Net Keras Model with adaptive depth.

    This is the single source of truth for the architecture.  Both ``UNet``
    and ``MaskedUNet3D`` delegate to this function.

    Parameters
    ----------
    input_size : tuple
        ``(D, H, W, channels)`` or ``(D, H, W, channels)`` shape.
    base_filters : int
        Filters in the first encoder block (doubled each level).
    min_size : int
        Minimum spatial size; controls maximum depth.
    dropout_layer : bool
        Whether to apply dropout in the bottleneck.
    dropout_rate : float
        Dropout probability.
    output_activation : callable
        Activation for the final Conv3D layer.
    logger_obj : logging.Logger or None
        Optional logger.

    Returns
    -------
    keras.Model
    """
    inputs = keras.layers.Input(shape=input_size)
    depth = compute_depth(input_size[:3], min_size)
    if logger_obj:
        logger_obj.info(
            "U-Net depth=%d (input=%s, min_size=%d)", depth, input_size, min_size,
        )

    skips = []
    x = inputs
    filters = base_filters

    # Encoder
    for _ in range(depth):
        x = keras.layers.Conv3D(filters, 3, activation='relu', padding='same')(x)
        x = keras.layers.Conv3D(filters, 3, activation='relu', padding='same')(x)
        skips.append(x)
        x = keras.layers.MaxPooling3D(2)(x)
        filters *= 2

    # Bottleneck
    x = keras.layers.Conv3D(filters, 3, activation='relu', padding='same')(x)
    x = keras.layers.Conv3D(filters, 3, activation='relu', padding='same')(x)
    if dropout_layer:
        x = keras.layers.Dropout(dropout_rate)(x)

    # Decoder
    for d in reversed(range(depth)):
        filters //= 2
        x = keras.layers.Conv3D(filters, 3, activation='relu', padding='same')(x)
        x = keras.layers.Conv3D(filters, 3, activation='relu', padding='same')(x)
        x = keras.layers.Conv3DTranspose(filters, 3, strides=2, padding='same')(x)
        x = keras.layers.concatenate([x, skips[d]], axis=-1)

    # Output (single channel)
    outputs = keras.layers.Conv3D(
        1, 3, padding='same', activation=output_activation, dtype='float32',
    )(x)

    return keras.Model(inputs=inputs, outputs=outputs)


# ---------------------------------------------------------------------------
# U-Net model
# ---------------------------------------------------------------------------

class UNet:
    """Adaptive 3D U-Net with configurable depth and output activation.

    The network depth is determined automatically from the spatial input size
    and ``min_size``, so the same class works for different grid resolutions.

    Parameters
    ----------
    base_filters : int
        Number of convolution filters in the first encoder block (doubled at
        each subsequent level).
    min_size : int
        Minimum spatial feature-map size; controls maximum depth.
    dropout_layer : bool
        Whether to apply dropout in the bottleneck.
    dropout_rate : float
        Dropout probability (used only when ``dropout_layer=True``).
    input_field : {'rho', 'delta'}
        Field type.  ``'rho'`` uses standard ReLU output; ``'delta'`` uses a
        shifted ReLU so that the output satisfies delta >= -1.
    norm_val : float
        Normalisation constant; determines the shift for the ``'delta'``
        activation.
    """

    def __init__(self, base_filters=16, min_size=4, dropout_layer=False,
                 dropout_rate=0.1, input_field='rho', norm_val=40):
        self.base_filters = base_filters
        self.min_size = min_size
        self.dropout_layer = dropout_layer
        self.dropout_rate = dropout_rate
        self.norm_val = norm_val
        self.output_activation = shifted_relu(norm_val) if input_field == 'delta' else tf.nn.relu
        self._logger = None

    def set_logger(self, log):
        """Attach an external logger (e.g. ``logging`` module)."""
        self._logger = log

    def prepare_model(self, input_size=(128, 128, 128, 2)):
        """Build and return the Keras ``Model``.

        Parameters
        ----------
        input_size : tuple
            ``(D, H, W, channels)`` shape of the input tensor.

        Returns
        -------
        keras.Model
        """
        return build_unet(
            input_size=input_size,
            base_filters=self.base_filters,
            min_size=self.min_size,
            dropout_layer=self.dropout_layer,
            dropout_rate=self.dropout_rate,
            output_activation=self.output_activation,
            logger_obj=self._logger,
        )


# ---------------------------------------------------------------------------
# Masked 3D U-Net
# ---------------------------------------------------------------------------

class MaskedUNet3D:
    """3D U-Net wrapper with mask-aware configuration.

    Delegates model construction to :func:`build_unet` and exposes the result
    via ``self.unet``.  The ``use_mask`` flag is stored for downstream scripts
    to query when deciding whether to apply mask-based post-processing.

    Parameters
    ----------
    input_size : tuple
        Spatial input shape including channels, e.g. ``(D, H, W, C)``.
    base_filters : int
        Filters in the first encoder block.
    min_size : int
        Minimum spatial size (controls depth).
    dropout_layer : bool
        Apply dropout in the bottleneck.
    dropout_rate : float
        Dropout probability.
    input_field : {'rho', 'delta'}
        Field type for output activation selection.
    norm_val : float
        Normalisation constant for the shifted-ReLU activation.
    use_mask : bool
        Whether mask-aware loss and post-processing should be used.
    logger : logging.Logger or None
        Optional logger for informational messages.
    """

    def __init__(self, input_size=(128, 128, 128, 1), base_filters=16,
                 min_size=4, dropout_layer=False, dropout_rate=0.1,
                 input_field='rho', norm_val=40, use_mask=False, logger=None):
        super().__init__()
        self.use_mask = use_mask
        self.norm_val = norm_val
        self.input_field = input_field
        self.output_activation = shifted_relu(norm_val) if input_field == 'delta' else tf.nn.relu
        self.unet = build_unet(
            input_size=input_size,
            base_filters=base_filters,
            min_size=min_size,
            dropout_layer=dropout_layer,
            dropout_rate=dropout_rate,
            output_activation=self.output_activation,
            logger_obj=logger,
        )


# ---------------------------------------------------------------------------
# Custom loss functions
# ---------------------------------------------------------------------------

class MaskedMSE(tf.keras.losses.Loss):
    """Mean Squared Error computed only over missing voxels (mask == 0).

    Parameters
    ----------
    mask : tf.Tensor or np.ndarray or None
        Binary mask where 1 = observed, 0 = missing.  Shape should be
        broadcastable to ``(batch, D, H, W, 1)``.  Pass ``None`` only when
        reconstructing from a saved config (mask will need to be set via
        ``set_mask`` before use).
    """

    def __init__(self, mask=None, **kwargs):
        super().__init__(**kwargs)
        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            self.mask = tf.Variable(
                mask, trainable=False, dtype=tf.float32, name='mse_mask',
            )
        else:
            self.mask = None

    def set_mask(self, mask):
        """Set or replace the mask after construction (e.g. when reloading)."""
        mask = tf.cast(mask, tf.float32)
        if self.mask is None:
            self.mask = tf.Variable(
                mask, trainable=False, dtype=tf.float32, name='mse_mask',
            )
        else:
            self.mask.assign(mask)

    def call(self, y_true, y_pred):
        if self.mask is None:
            raise RuntimeError(
                "MaskedMSE.mask is None -- call set_mask() before training "
                "or pass mask= to the constructor."
            )
        diff = tf.square(y_true - y_pred)
        inv_mask = 1 - self.mask
        masked_diff = diff * inv_mask
        denom = tf.reduce_sum(inv_mask) + 1e-8
        return tf.reduce_sum(masked_diff) / denom

    def get_config(self):
        config = super().get_config()
        # mask is a tf.Variable and cannot be serialised to JSON; it must be
        # re-supplied when loading via custom_objects + set_mask().
        config.update({"mask": None})
        return config

    @classmethod
    def from_config(cls, config):
        """Reconstruct from config.  The mask must be set separately."""
        config.pop("mask", None)
        return cls(mask=None, **config)


class MaskedGradientLoss(tf.keras.losses.Loss):
    """Gradient-matching loss near the boundaries of masked holes.

    The loss penalises differences in spatial gradients between prediction and
    truth in a dilated region around the holes, promoting smooth transitions.

    Parameters
    ----------
    mask : array-like or None
        Binary mask (0 = hole, 1 = valid).  Pass ``None`` only when
        reconstructing from config.
    dilation_iter : int
        Number of dilation passes to expand the control region around holes.
    """

    def __init__(self, mask=None, dilation_iter=2, **kwargs):
        super().__init__(**kwargs)
        self.dilation_iter = dilation_iter
        self._masks_ready = False

        if mask is not None:
            self._init_masks(mask)

    def _init_masks(self, mask):
        mask_tensor = prepare_mask_tensor(mask)
        hole_mask = 1.0 - mask_tensor
        dilated_hole = hole_mask
        for _ in range(self.dilation_iter):
            dilated_hole = tf.nn.max_pool3d(
                dilated_hole,
                ksize=[1, 3, 3, 3, 1],
                strides=[1, 1, 1, 1, 1],
                padding='SAME',
            )

        dh = tf.squeeze(dilated_hole, axis=0)
        self.mask_x = tf.convert_to_tensor(dh[1:, :, :, :][None, ...], dtype=tf.float32)
        self.mask_y = tf.convert_to_tensor(dh[:, 1:, :, :][None, ...], dtype=tf.float32)
        self.mask_z = tf.convert_to_tensor(dh[:, :, 1:, :][None, ...], dtype=tf.float32)

        self.denom_x = tf.constant(tf.reduce_sum(self.mask_x) + 1e-8, dtype=tf.float32)
        self.denom_y = tf.constant(tf.reduce_sum(self.mask_y) + 1e-8, dtype=tf.float32)
        self.denom_z = tf.constant(tf.reduce_sum(self.mask_z) + 1e-8, dtype=tf.float32)
        self._masks_ready = True

    def set_mask(self, mask):
        """Set or replace the mask (e.g. after reloading from config)."""
        self._init_masks(mask)

    def call(self, y_true, y_pred):
        if not self._masks_ready:
            raise RuntimeError(
                "MaskedGradientLoss masks not initialised -- call set_mask() first."
            )
        gx_t, gy_t, gz_t = compute_gradient(y_true)
        gx_p, gy_p, gz_p = compute_gradient(y_pred)

        term_x = tf.reduce_sum(tf.square(gx_t - gx_p) * self.mask_x) / self.denom_x
        term_y = tf.reduce_sum(tf.square(gy_t - gy_p) * self.mask_y) / self.denom_y
        term_z = tf.reduce_sum(tf.square(gz_t - gz_p) * self.mask_z) / self.denom_z
        return term_x + term_y + term_z

    def get_config(self):
        config = super().get_config()
        config.update({"mask": None, "dilation_iter": self.dilation_iter})
        return config

    @classmethod
    def from_config(cls, config):
        config.pop("mask", None)
        return cls(mask=None, **config)


class MaskedMSEWithGradient(tf.keras.losses.Loss):
    """Combined masked MSE + weighted gradient-matching loss.

    Parameters
    ----------
    mask : array-like or None
        Binary mask (0 = hole, 1 = valid).
    gradient_weight : float
        Relative weight of the gradient loss term.
    """

    def __init__(self, mask=None, gradient_weight=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mse_loss_fn = MaskedMSE(mask)
        self.grad_loss_fn = MaskedGradientLoss(mask)
        self.gradient_weight = gradient_weight

    def set_mask(self, mask):
        """Forward mask to both sub-losses."""
        self.mse_loss_fn.set_mask(mask)
        self.grad_loss_fn.set_mask(mask)

    def call(self, y_true, y_pred):
        mse_loss = self.mse_loss_fn(y_true, y_pred)
        grad_loss = self.grad_loss_fn(y_true, y_pred)
        return mse_loss + self.gradient_weight * grad_loss

    def get_config(self):
        config = super().get_config()
        config.update({"gradient_weight": self.gradient_weight})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(mask=None, **config)


# ---------------------------------------------------------------------------
# Deprecated alias
# ---------------------------------------------------------------------------

def MaskedInpaintingUNet(*args, **kwargs):
    """Deprecated: use ``MaskedUNet3D`` instead."""
    import warnings
    warnings.warn(
        "MaskedInpaintingUNet is deprecated, use MaskedUNet3D instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return MaskedUNet3D(*args, **kwargs)

