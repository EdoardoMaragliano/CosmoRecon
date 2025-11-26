# datahandler.py
import os
import logging
import numpy as np
import tensorflow as tf

from CosmoFiller.utils.loggers import setup_logger

# -------------------------
# Module logger
# -------------------------
logger = setup_logger(__name__)


def parse_fn(obs_path, true_path, mask_path=None, single_mask=None, use_mask=True,
             field_size=128, norm_val=40):
    """
    Parse input, target, and optional mask for a single 3D field sample.
    
    Parameters:
        obs_path: str, path to observed field (.npy)
        true_path: str, path to true field (.npy)
        mask_path: str, path to mask (.npy) for this sample
        single_mask: np.array, optional mask shared by all samples
        use_mask: bool, whether to use mask channel
        field_size: int, spatial size of the cube (NxNxN)
        norm_val: float, normalization factor for fields
        
    Returns:
        tuple: (x, y_true, mask) if use_mask else (x, y_true)
    """
    obs = tf.numpy_function(lambda f: np.load(f).astype(np.float32)/norm_val,
                            [obs_path], tf.float32)
    true = tf.numpy_function(lambda f: np.load(f).astype(np.float32)/norm_val,
                             [true_path], tf.float32)
    obs = tf.expand_dims(obs, axis=-1)
    true = tf.expand_dims(true, axis=-1)

    if use_mask:
        if single_mask is not None:
            mask = tf.convert_to_tensor(single_mask, dtype=tf.float32)
            logging.debug(f"Using single mask with shape: {mask.shape}")
            mask = tf.expand_dims(mask, axis=-1)
            logging.debug(f"Expanded single mask shape: {mask.shape}")
        else:
            mask = tf.numpy_function(lambda f: np.load(f), [mask_path], tf.float32)
            mask = tf.expand_dims(mask, axis=-1)
            logging.debug(f"Loaded mask from {mask_path} with shape: {mask.shape}")
            

        x = tf.concat([obs, mask], axis=-1)
        x.set_shape((field_size, field_size, field_size, 2))
        true.set_shape((field_size, field_size, field_size, 1))
        mask.set_shape((field_size, field_size, field_size, 1))
        logging.debug(f"Input concatenated (obs + mask), shape: {x.shape}")
        return (x, true, mask)
    else:
        x = obs
        x.set_shape((field_size, field_size, field_size, 1))
        true.set_shape((field_size, field_size, field_size, 1))
        logging.debug(f"Single channel input (obs only), shape: {x.shape}")
        return (x, true)


def create_dataset(x_files, y_files, mask_files=None, single_mask=None, batch_size=16,
                   shuffle=True, use_mask=True, repeat=False, drop_remainder=False,
                   field_size=128, norm_val=40):
    """
    Create a tf.data.Dataset from file lists with optional mask channel.
    
    Parameters:
        x_files, y_files: lists of file paths
        mask_files: optional list of masks
        single_mask: optional shared mask
        batch_size: int
        shuffle: bool
        use_mask: bool
        repeat: bool
        drop_remainder: bool
        field_size: int
        norm_val: float
        
    Returns:
        tf.data.Dataset
    """
    logging.info(f"Creating dataset with {len(x_files)} samples. Use mask: {use_mask}")
    
    if use_mask and single_mask is not None:
        dataset = tf.data.Dataset.from_tensor_slices((x_files, y_files))
        logging.debug("Dataset created from x_files and y_files only. Mapping with single_mask.")
        dataset = dataset.map(lambda x, y: parse_fn(x, y, single_mask=single_mask,
                                                    use_mask=True,
                                                    field_size=field_size,
                                                    norm_val=norm_val),
                              num_parallel_calls=tf.data.AUTOTUNE)
    elif use_mask and mask_files is not None:
        dataset = tf.data.Dataset.from_tensor_slices((x_files, y_files, mask_files))
        logging.debug("Dataset created from x_files, y_files, and mask_files. Mapping with mask_path.")
        dataset = dataset.map(lambda x, y, m: parse_fn(x, y, mask_path=m, use_mask=True,
                                                       field_size=field_size,
                                                       norm_val=norm_val),
                              num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = tf.data.Dataset.from_tensor_slices((x_files, y_files))
        logging.debug("Dataset created from x_files and y_files only. Mapping without mask.")
        dataset = dataset.map(lambda x, y: parse_fn(x, y, use_mask=False,
                                                    field_size=field_size,
                                                    norm_val=norm_val),
                              num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(x_files), reshuffle_each_iteration=True)
        logging.debug("Dataset shuffled")
    if repeat:
        dataset = dataset.repeat()
        logging.debug("Dataset repeated indefinitely")
        
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder).prefetch(tf.data.AUTOTUNE)
    logging.debug(f"Dataset batched with batch_size={batch_size}, drop_remainder={drop_remainder}")
    
    return dataset
