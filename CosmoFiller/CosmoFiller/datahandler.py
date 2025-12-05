# datahandler.py

"""
This module provides functions to create TensorFlow datasets for 3D cosmological field inpainting tasks.
It supports loading observed fields, true fields, and optional masks from .npy files, normalizing them,
and preparing them for training or evaluation with batching, shuffling, and prefetching.
"""

import os
import logging
logger = logging.getLogger(__name__)
import numpy as np
import tensorflow as tf

from CosmoFiller.utils.loggers import setup_logger

# -------------------------
# Module logger
# -------------------------
logger = setup_logger(__name__)


def parse_fn(obs_path, true_path, mask=None,
             field_size=128, norm_val=40, channels=1):
    """
    Load a single pair (observed, true) and optionally apply a global mask.

    Parameters
    ----------
    obs_path : str
        Path to observed volume (.npy)
    true_path : str
        Path to true volume (.npy)
    mask : np.ndarray or None
        Single 3D mask shared by all samples (float32 0/1)
    channels : int
        1  -> input = obs * mask
        2  -> input = concat(obs * mask, mask)

    Returns
    -------
    (x, y)  with shapes:
        x : (N,N,N,channels)
        y : (N,N,N,1)
    """
    # Load volumes
    obs = tf.numpy_function(lambda f: np.load(f).astype(np.float32)/norm_val,
                            [obs_path], tf.float32)
    true = tf.numpy_function(lambda f: np.load(f).astype(np.float32)/norm_val,
                             [true_path], tf.float32)

    # Add channel dimension
    obs = tf.expand_dims(obs, axis=-1)     # (N,N,N,1)
    true = tf.expand_dims(true, axis=-1)   # (N,N,N,1)

    # Static shapes
    obs.set_shape((field_size, field_size, field_size, 1))
    true.set_shape((field_size, field_size, field_size, 1))

    # ------ Apply mask if provided ------
    if mask is not None:
        mask_tf = tf.convert_to_tensor(mask, dtype=tf.float32)
        mask_tf = tf.expand_dims(mask, axis=-1)
        mask_tf.set_shape((field_size, field_size, field_size, 1))

        masked_obs = obs * mask_tf
        logger.debug(f"Applied mask to observed data, shape: {obs.shape}")    

        if channels==1:
            x = masked_obs

        elif channels==2:
            x = tf.concat([masked_obs, mask_tf], axis=-1)

        else:
            raise ValueError("channels must be 1 or 2")
    
    else:
        x = obs

        if channels !=1:
            raise ValueError("channels=2 reuires a mask")
        
    return x, true

def create_dataset(x_files, y_files,
                   batch_size=16,
                   shuffle=True,
                   repeat=False,
                   drop_remainder=False,
                   field_size=128,
                   norm_val=40,
                   mask=None,
                   channels=1):
    """
    Create a tf.data.Dataset with optional single global mask.

    Parameters
    ----------
    x_files, y_files : list of str
        Paths to observed and true volumes.
    mask : np.ndarray or None
        Single shared mask (N,N,N).
    channels : int
        1 = obs*mask
        2 = concat(obs*mask, mask)

    Returns
    -------
    tf.data.Dataset yielding (x_batch, y_batch)
    """

    logger.info(f"Creating dataset with {len(x_files)} samples, channels={channels}, mask={mask is not None}")

    dataset = tf.data.Dataset.from_tensor_slices((x_files, y_files))

    dataset = dataset.map(
        lambda x, y: parse_fn(
            x, y,
            field_size=field_size,
            norm_val=norm_val,
            mask=mask,
            channels=channels
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(x_files), reshuffle_each_iteration=True)
        logger.debug("Dataset shuffled")

    if repeat:
        dataset = dataset.repeat()
        logger.debug("Dataset repeated indefinitely")

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    logger.debug(f"Dataset batched with batch_size={batch_size}, drop_remainder={drop_remainder}")

    return dataset
