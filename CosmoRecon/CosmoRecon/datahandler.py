"""
Data handler for 3D cosmological field reconstruction.

Provides functions to build ``tf.data.Dataset`` pipelines that load observed
and true density fields from ``.npy`` files, apply optional binary masks,
normalise values, and prepare batches for training or evaluation.
"""

import numpy as np
import tensorflow as tf
from typing import Callable, List, Optional

from CosmoRecon.utils.loggers import setup_logger

logger = setup_logger(__name__)


def _make_parse_fn(mask_tensor: Optional[tf.Tensor], field_size: int,
                   norm_val: float, channels: int) -> Callable:
    """Return a parse function with the mask pre-converted to a tensor.

    This avoids calling ``tf.convert_to_tensor`` on every sample inside the
    ``tf.data.Dataset.map()`` call.

    Parameters
    ----------
    mask_tensor : tf.Tensor or None
        Pre-converted mask tensor of shape ``(N, N, N, 1)`` or ``None``.
    field_size : int
        Spatial grid size *N*.
    norm_val : float
        Normalisation divisor.
    channels : int
        1 = obs * mask;  2 = concat(obs * mask, mask).

    Returns
    -------
    callable
        Function with signature ``(obs_path, true_path) -> (x, y)``.
    """
    def parse_fn(obs_path, true_path):
        obs = tf.numpy_function(
            lambda f: np.load(f).astype(np.float32) / norm_val,
            [obs_path], tf.float32,
        )
        true = tf.numpy_function(
            lambda f: np.load(f).astype(np.float32) / norm_val,
            [true_path], tf.float32,
        )

        obs = tf.expand_dims(obs, axis=-1)
        true = tf.expand_dims(true, axis=-1)

        obs.set_shape((field_size, field_size, field_size, 1))
        true.set_shape((field_size, field_size, field_size, 1))

        if mask_tensor is not None:
            masked_obs = obs * mask_tensor
            if channels == 1:
                x = masked_obs
            else:
                x = tf.concat([masked_obs, mask_tensor], axis=-1)
        else:
            x = obs

        return x, true

    return parse_fn


def create_dataset(x_files: List[str], y_files: List[str], batch_size: int = 16,
                   shuffle: bool = True, repeat: bool = False,
                   drop_remainder: bool = False, field_size: int = 128,
                   norm_val: float = 40, mask: Optional[np.ndarray] = None,
                   channels: int = 1) -> tf.data.Dataset:
    """Create a ``tf.data.Dataset`` with optional single global mask.

    Parameters
    ----------
    x_files, y_files : list[str]
        Paths to observed and true volumes.
    batch_size : int
        Batch size.
    shuffle : bool
        Shuffle the dataset each epoch.
    repeat : bool
        Repeat the dataset indefinitely (useful for ``steps_per_epoch``).
    drop_remainder : bool
        Drop the last incomplete batch.
    field_size : int
        Spatial grid size *N*.
    norm_val : float
        Normalisation divisor.
    mask : np.ndarray or None
        Single shared mask ``(N, N, N)``.
    channels : int
        1 = obs * mask, 2 = concat(obs * mask, mask).

    Returns
    -------
    tf.data.Dataset
        Yields ``(x_batch, y_batch)`` tensors.

    Raises
    ------
    ValueError
        If ``channels=2`` is requested without providing a mask.
    """
    if channels not in (1, 2):
        raise ValueError(f"channels must be 1 or 2, got {channels}")
    if channels == 2 and mask is None:
        raise ValueError("channels=2 requires a mask")

    logger.info(
        "Creating dataset: %d samples, channels=%d, mask=%s",
        len(x_files), channels, mask is not None,
    )

    # Pre-convert mask to tensor once (not per-sample)
    mask_tensor = None
    if mask is not None:
        mask_tensor = tf.constant(mask, dtype=tf.float32)
        mask_tensor = tf.expand_dims(mask_tensor, axis=-1)
        mask_tensor.set_shape((field_size, field_size, field_size, 1))

    parse_fn = _make_parse_fn(mask_tensor, field_size, norm_val, channels)

    dataset = tf.data.Dataset.from_tensor_slices((x_files, y_files))
    dataset = dataset.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(x_files), reshuffle_each_iteration=True,
        )

    if repeat:
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
