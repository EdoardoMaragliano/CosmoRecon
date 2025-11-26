#!/usr/bin/env python3
"""
train_inpainting_plus.py
Versione FIXATA "PLUS" in un unico file per l'allenamento di una 3D U-Net
- Gestione robusta dei file
- Data pipeline ottimizzata (cache, prefetch, AUTOTUNE)
- Mixed precision (opzionale)
- Mask handling (single mask o per-sample)
- Loss mascherata implementata in Custom Model
- Multi-GPU via MirroredStrategy
- Salvataggio modello ogni N epoche tramite callback custom
Author: adattato per te
Date: versione FIXATA
"""

import os
import re
import glob
import argparse
import logging
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from datetime import datetime
from inpainting import UNet, MaskedInpaintingModel
from checkpoints import SaveEveryNEpoch

# -----------------------------
# Utility: natural sort for filenames
# -----------------------------
def natural_sort(l):
    """Sort strings with embedded numbers in human order."""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key=alphanum_key)




# -----------------------------
# Arg parser (robust flags)
# -----------------------------
parser = argparse.ArgumentParser(description='Train 3D U-Net for inpainting (fixed PLUS).')
parser.add_argument('--param_file', type=str, help='Path to JSON parameter file')
parser.add_argument('--obs_dir', type=str, required=True, help='Directory with observed .npy fields')
parser.add_argument('--true_dir', type=str, required=True, help='Directory with true .npy fields')
parser.add_argument('--mask_dir', type=str, default=None, help='Directory with mask .npy files (optional)')
parser.add_argument('--obs_pattern', type=str, default='*.npy', help='Glob pattern for observed fields')
parser.add_argument('--true_pattern', type=str, default='*.npy', help='Glob pattern for true fields')
parser.add_argument('--input_field', type=str, choices=['rho', 'delta'], default='rho',
                    help='Type of input field')
parser.add_argument('--output_dir', type=str, default='output_products', help='Directory to store outputs')
parser.add_argument('--use_mask', action='store_true', help='Enable mask usage (default: False)')
parser.add_argument('--field_size', type=int, default=128, help='Size of input fields (NxNxN)')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--repeat_dataset', action='store_true', help='Repeat dataset indefinitely for training')
parser.add_argument('--drop_remainder', action='store_true', help='Drop remainder in batching for consistent batch sizes')
parser.add_argument('--save_freq', type=int, default=10, help='Save model every N epochs')
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--log_file', type=str, default='training.log', help='Log filename')
parser.add_argument('--debug', action='store_true', help='Enable debug logs')
parser.add_argument('--density_normalization', type=float, default=40.0)
parser.add_argument('--min_mock_idx', type=int, default=None)
parser.add_argument('--max_mock_idx', type=int, default=None)
parser.add_argument('--mixed_precision', action='store_true', help='Enable mixed precision (faster on modern GPUs)')
parser.add_argument('--base_filters', type=int, default=16, help='Base number of conv filters')
parser.add_argument('--min_size', type=int, default=4, help='Min feature map size for adaptive depth')
parser.add_argument('--dropout', action='store_true', help='Use dropout in bottleneck')
parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate')
parser.add_argument('--global_clipnorm', type=float, default=1.0, help='Gradient clipnorm (sensible default)')
args = parser.parse_args()

# Load params from JSON if provided (overrides CLI where applicable)
if args.param_file:
    with open(args.param_file, 'r') as f:
        params = json.load(f)
    for k, v in params.items():
        if hasattr(args, k):
            setattr(args, k, v)

# -----------------------------
# Prepare directories & logging
# -----------------------------
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'store_models'), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'output_data'), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'losses'), exist_ok=True)

# configure logging
log_level = logging.DEBUG if args.debug else logging.INFO
logging.basicConfig(
    level=log_level,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(args.output_dir, args.log_file)),
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)
logging.info("Starting training script")
logging.info(f"Arguments: {args}")

# Mixed precision
if args.mixed_precision:
    try:
        from keras.mixed_precision import experimental as mixed_precision
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        logging.info("Mixed precision enabled (mixed_float16).")
    except Exception as e:
        logging.warning("Mixed precision not enabled: %s", e)

# -----------------------------
# Collect and sort files robustly
# -----------------------------
x_files = natural_sort(glob.glob(os.path.join(args.obs_dir, args.obs_pattern)))
y_files = natural_sort(glob.glob(os.path.join(args.true_dir, args.true_pattern)))
mask_files = natural_sort(glob.glob(os.path.join(args.mask_dir, '*.npy'))) if args.mask_dir else []

# Optionally subset by index range
if args.min_mock_idx is not None or args.max_mock_idx is not None:
    lo = args.min_mock_idx if args.min_mock_idx is not None else 0
    hi = args.max_mock_idx if args.max_mock_idx is not None else None
    x_files = x_files[lo:hi]
    y_files = y_files[lo:hi]
    if mask_files:
        mask_files = mask_files[lo:hi]

logging.info(f"Found {len(x_files)} observed files and {len(y_files)} true files.")
if args.use_mask:
    logging.info(f"Found {len(mask_files)} mask files.")

# Basic checks
if len(x_files) == 0 or len(y_files) == 0:
    raise RuntimeError("No input files found. Check obs_dir/true_dir and patterns.")

if len(x_files) != len(y_files):
    raise RuntimeError("Number of observed files and true files mismatch!")

# Handle mask modes
use_mask = bool(args.use_mask)
use_single_mask = False
single_mask = None
if use_mask:
    if len(mask_files) == 0:
        logging.warning("use_mask=True but no mask files found -> turning use_mask off.")
        use_mask = False
    elif len(mask_files) == 1:
        # load single mask now (numpy) for speed
        single_mask = np.load(mask_files[0]).astype(np.float32)
        use_single_mask = True
        logging.info("Using a single mask for all samples.")
    else:
        if len(mask_files) != len(x_files):
            raise RuntimeError("Number of masks does not match number of samples.")
        logging.info("Using individual masks for each sample.")

# -----------------------------
# Split dataset (preserve pairing)
# -----------------------------
if use_mask and not use_single_mask:
    x_train, x_valid, y_train, y_valid, mask_train, mask_valid = train_test_split(
        x_files, y_files, mask_files, test_size=0.2, shuffle=False)
else:
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_files, y_files, test_size=0.2, shuffle=False)
    mask_train = mask_valid = None

logging.info(f"Training samples: {len(x_train)}, Validation samples: {len(x_valid)}")

# -----------------------------
# Helper numpy loader functions for tf.numpy_function
# -----------------------------
def _np_load_field(path, norm_val, field_flag):
    """
    path: bytes -> decode
    norm_val: np.float32
    field_flag: np.int32 (0 -> rho, 1 -> delta)
    """
    p = path.decode('utf-8')
    arr = np.load(p).astype(np.float32)
    if int(field_flag) == 0:
        # rho: normalize
        arr = arr / float(norm_val)
    # delta: do not normalize targets (we keep values as-is)
    return arr

def _np_load_mask(path):
    p = path.decode('utf-8')
    arr = np.load(p).astype(np.float32)
    return arr

# -----------------------------
# Dataset parsing & creation
# -----------------------------
AUTOTUNE = tf.data.AUTOTUNE
field_size = args.field_size

def parse_with_mask(obs_path, true_path, mask_path):
    # obs_path, true_path, mask_path are tf.string tensors
    obs = tf.numpy_function(_np_load_field,
                            [obs_path, np.float32(args.density_normalization), np.int32(0 if args.input_field=='rho' else 1)],
                            tf.float32)
    true = tf.numpy_function(_np_load_field,
                             [true_path, np.float32(args.density_normalization), np.int32(0 if args.input_field=='rho' else 1)],
                             tf.float32)
    mask = tf.numpy_function(_np_load_mask, [mask_path], tf.float32)

    # set static shapes so TF graph knows shapes
    obs.set_shape((field_size, field_size, field_size))
    true.set_shape((field_size, field_size, field_size))
    mask.set_shape((field_size, field_size, field_size))

    obs = tf.expand_dims(obs, axis=-1)
    true = tf.expand_dims(true, axis=-1)
    mask = tf.expand_dims(mask, axis=-1)

    x = tf.concat([obs, mask], axis=-1)
    # set shape explicitly: channels 2 (obs + mask)
    x.set_shape((field_size, field_size, field_size, 2))

    return x, true, mask

def parse_single_mask(obs_path, true_path):
    obs = tf.numpy_function(_np_load_field,
                            [obs_path, np.float32(args.density_normalization), np.int32(0 if args.input_field=='rho' else 1)],
                            tf.float32)
    true = tf.numpy_function(_np_load_field,
                             [true_path, np.float32(args.density_normalization), np.int32(0 if args.input_field=='rho' else 1)],
                             tf.float32)

    obs.set_shape((field_size, field_size, field_size))
    true.set_shape((field_size, field_size, field_size))

    obs = tf.expand_dims(obs, axis=-1)
    true = tf.expand_dims(true, axis=-1)

    # single_mask is a numpy array -> convert to tensor
    mask_tensor = tf.convert_to_tensor(single_mask, dtype=tf.float32)
    mask_tensor = tf.reshape(mask_tensor, (field_size, field_size, field_size, 1))

    x = tf.concat([obs, mask_tensor], axis=-1)
    x.set_shape((field_size, field_size, field_size, 2))

    return x, true, mask_tensor

def parse_no_mask(obs_path, true_path):
    obs = tf.numpy_function(_np_load_field,
                            [obs_path, np.float32(args.density_normalization), np.int32(0 if args.input_field=='rho' else 1)],
                            tf.float32)
    true = tf.numpy_function(_np_load_field,
                             [true_path, np.float32(args.density_normalization), np.int32(0 if args.input_field=='rho' else 1)],
                             tf.float32)

    obs.set_shape((field_size, field_size, field_size))
    true.set_shape((field_size, field_size, field_size))
    obs = tf.expand_dims(obs, axis=-1)
    true = tf.expand_dims(true, axis=-1)

    x = obs
    x.set_shape((field_size, field_size, field_size, 1))
    return x, true

def create_dataset(x_list, y_list, mask_list=None, single_mask_arg=None,
                   batch_size=8, shuffle=True, repeat=False, drop_remainder=False):
    if mask_list is not None and single_mask_arg is None:
        ds = tf.data.Dataset.from_tensor_slices((x_list, y_list, mask_list))
        ds = ds.map(lambda a,b,c: parse_with_mask(a,b,c), num_parallel_calls=AUTOTUNE)
    elif single_mask_arg is not None:
        ds = tf.data.Dataset.from_tensor_slices((x_list, y_list))
        ds = ds.map(lambda a,b: parse_single_mask(a,b), num_parallel_calls=AUTOTUNE)
    else:
        ds = tf.data.Dataset.from_tensor_slices((x_list, y_list))
        ds = ds.map(lambda a,b: parse_no_mask(a,b), num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(x_list), 512))

    # cache optional for small datasets (safe: use disk-backed caching by providing filename if needed)
    # ds = ds.cache()

    if repeat:
        ds = ds.repeat()

    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(AUTOTUNE)
    return ds

train_dataset = create_dataset(
    x_train, y_train,
    mask_list=mask_train if (use_mask and not use_single_mask) else None,
    single_mask_arg=single_mask if use_single_mask else None,
    batch_size=args.batch_size,
    shuffle=True,
    repeat=args.repeat_dataset,
    drop_remainder=args.drop_remainder
)

val_dataset = create_dataset(
    x_valid, y_valid,
    mask_list=mask_valid if (use_mask and not use_single_mask) else None,
    single_mask_arg=single_mask if use_single_mask else None,
    batch_size=args.batch_size,
    shuffle=False,
    repeat=False,
    drop_remainder=False
)

# debug: print sample shapes (only in eager mode)
if args.debug:
    for batch in train_dataset.take(1):
        if use_mask:
            x_sample, y_sample, mask_sample = batch
            logging.debug("Sample shapes (x,y,mask): %s %s %s", x_sample.shape, y_sample.shape, mask_sample.shape)
        else:
            x_sample, y_sample = batch
            logging.debug("Sample shapes (x,y): %s %s", x_sample.shape, y_sample.shape)

# -----------------------------
# Build model inside strategy scope
# -----------------------------
strategy = tf.distribute.MirroredStrategy()
logging.info(f"Number of devices: {strategy.num_replicas_in_sync}")

keras.backend.clear_session()

with strategy.scope():
    mask_channels = 2 if use_mask else 1
    input_size = (field_size, field_size, field_size, mask_channels)
    logging.info(f"Preparing UNet with input_size={input_size}")

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
    masked_model = MaskedInpaintingModel(inputs=base_unet.input, outputs=base_unet.output, use_mask=use_mask)

    # Optimizer with reasonable clipnorm
    optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)
    # apply clipnorm as requested
    if args.global_clipnorm and args.global_clipnorm > 0:
        optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate, clipnorm=float(args.global_clipnorm))

    # compile with a metric (MSE on full volume is a coarse metric; loss uses mask inside train_step)
    masked_model.compile(optimizer=optimizer, metrics=[keras.metrics.MeanSquaredError(name='mse')])

    logging.info("Model built inside strategy scope.")
    logging.info("Model summary:")
    base_unet.summary(print_fn=lambda s: logging.info(s))

# -----------------------------
# Callbacks
# -----------------------------
checkpoint_dir = os.path.join(args.output_dir, 'store_models')
os.makedirs(checkpoint_dir, exist_ok=True)

savecb = SaveEveryNEpoch(save_dir=checkpoint_dir, period=args.save_freq, save_best_only=False)
tensorboard_cb = keras.callbacks.TensorBoard(log_dir=os.path.join(args.output_dir, 'logs'), histogram_freq=1)
earlystop_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=1, restore_best_weights=True)
csv_logger = keras.callbacks.CSVLogger(os.path.join(args.output_dir, 'history.csv'), append=False)

callbacks = [savecb, tensorboard_cb, earlystop_cb, csv_logger]

# -----------------------------
# Training: steps_per_epoch handling when dataset repeats
# -----------------------------
if args.repeat_dataset:
    steps_per_epoch = int(np.ceil(len(x_train) / args.batch_size))
    logging.info(f"repeat_dataset=True -> using steps_per_epoch={steps_per_epoch}")
else:
    steps_per_epoch = None

logging.info("Starting training with fit()...")
start_time = datetime.now()

history = masked_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=args.epochs,
    callbacks=callbacks,
    steps_per_epoch=steps_per_epoch,
    verbose=2
)

end_time = datetime.now()
logging.info(f"Training finished. Duration: {end_time - start_time}")

# -----------------------------
# Save loss arrays
# -----------------------------
np.save(os.path.join(args.output_dir, 'losses', 'loss.npy'), np.array(history.history.get('loss', [])))
np.save(os.path.join(args.output_dir, 'losses', 'val_loss.npy'), np.array(history.history.get('val_loss', [])))
logging.info("Saved loss arrays.")

# -----------------------------
# Evaluation & saving predictions (batch_size=1)
# -----------------------------
logging.info("Creating evaluation dataset (batch_size=1) for predictions...")
val_dataset_eval = create_dataset(
    x_valid, y_valid,
    mask_list=mask_valid if (use_mask and not use_single_mask) else None,
    single_mask_arg=single_mask if use_single_mask else None,
    batch_size=1,
    shuffle=False,
    repeat=False,
    drop_remainder=False
)

logging.info("Running predictions on validation set...")
predictions = masked_model.predict(val_dataset_eval, verbose=1)

# predictions shape: (N, Nx, Ny, Nz, 1)
outdir = os.path.join(args.output_dir, 'output_data')
os.makedirs(outdir, exist_ok=True)
for i, pred in enumerate(predictions):
    # denormalize if we normalized rho
    pred_field = pred[..., 0]
    if args.input_field == 'rho':
        pred_field = pred_field * args.density_normalization
    # save
    fname = os.path.join(outdir, f'pred_field_{i:04d}.npy')
    np.save(fname, pred_field.astype(np.float32))
    logging.info("Saved %s", fname)

logging.info("All predictions saved.")
logging.info("Script finished successfully.")
