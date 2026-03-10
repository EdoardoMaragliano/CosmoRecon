#!/usr/bin/env python3
"""Unified training script for CosmoRecon 3D U-Net reconstruction.

Supports three loss modes via ``--loss_type``:
  - ``mse``            : standard MSE (no mask weighting)
  - ``masked_mse``     : MSE restricted to missing voxels
  - ``masked_gradient`` : masked MSE + gradient-matching loss

All arguments can be overridden via a JSON parameter file (``--param_file``).
"""

import argparse
import glob
import json
import logging
import os
import random
import sys
from datetime import datetime
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Ensure project root is importable (fallback for non-installed use)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from CosmoRecon.models import (
    UNet,
    MaskedUNet3D,
    MaskedMSE,
    MaskedMSEWithGradient,
)
from CosmoRecon.checkpoints import SaveEveryNEpoch
from CosmoRecon.datahandler import create_dataset
from CosmoRecon.utils.gpu import configure_gpus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_gpu_indices(s: Optional[str]):
    """Parse a comma-separated string of GPU indices, e.g. '0,1,3'."""
    if s is None:
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _load_json_params(args: argparse.Namespace, json_path: str, logger: logging.Logger):
    """Override argparse values with entries from a JSON file.

    Keys not matching any argparse argument trigger a warning.
    """
    with open(json_path, 'r') as f:
        params = json.load(f)
    for key, value in params.items():
        if hasattr(args, key):
            setattr(args, key, value)
        else:
            logger.warning("Unknown JSON key '%s' ignored.", key)


def _set_random_seeds(seed: int):
    """Set random seeds for reproducibility across Python, NumPy, and TF."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train CosmoRecon 3D U-Net for field reconstruction",
    )
    # Mode
    parser.add_argument(
        '--loss_type', type=str, default='mse',
        choices=['mse', 'masked_mse', 'masked_gradient'],
        help='Loss function to use for training',
    )

    # Paths
    parser.add_argument('--param_file', type=str, help='Path to JSON parameter file')
    parser.add_argument('--obs_dir', type=str, help='Directory with observed .npy fields')
    parser.add_argument('--true_dir', type=str, help='Directory with true .npy fields')
    parser.add_argument('--mask_dir', type=str, default=None,
                        help='Directory with mask .npy files')
    parser.add_argument('--obs_basefile', type=str, default='*.npy')
    parser.add_argument('--true_basefile', type=str, default='*.npy')
    parser.add_argument('--input_field', type=str, default='rho',
                        choices=['rho', 'delta'])
    parser.add_argument('--output_dir', type=str, default='output_products')

    # Data
    parser.add_argument('--use_mask', action='store_true')
    parser.add_argument('--field_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--repeat_dataset', action='store_true')
    parser.add_argument('--drop_remainder', action='store_true')
    parser.add_argument('--min_mock_idx', type=int, default=None)
    parser.add_argument('--max_mock_idx', type=int, default=None)

    # Model
    parser.add_argument('--base_filters', type=int, default=16)
    parser.add_argument('--min_size', type=int, default=4)
    parser.add_argument('--dropout', action='store_true')
    parser.add_argument('--dropout_rate', type=float, default=0.1)

    # Optimiser
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--global_clipnorm', type=float, default=1e-2)

    # Loss params
    parser.add_argument('--gradient_weight', type=float, default=0.1,
                        help='Weight of gradient loss term (masked_gradient mode)')
    parser.add_argument('--density_normalization', type=float, default=40.0)

    # Training
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--save_freq', type=int, default=10)

    # Infrastructure
    parser.add_argument('--log_file', type=str, default='training.log')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--gpu_indices', type=str, default=None,
                        help='Comma-separated GPU device indices (e.g. "0,1")')
    parser.add_argument('--memory_growth', action='store_true',
                        help='Enable GPU memory growth')

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    start_time = datetime.now()

    parser = build_parser()
    args = parser.parse_args()

    # -- GPU configuration (before any TF operations) -----------------------
    configure_gpus(
        device_indices=_parse_gpu_indices(args.gpu_indices),
        memory_growth=args.memory_growth,
    )

    # -- Preliminary logger for JSON loading --------------------------------
    _pre_logger = logging.getLogger("train.preload")

    # -- JSON parameter override --------------------------------------------
    if args.param_file:
        _load_json_params(args, args.param_file, _pre_logger)

    # -- Seed ---------------------------------------------------------------
    if args.seed is not None:
        _set_random_seeds(args.seed)

    # -- Validation ---------------------------------------------------------
    if args.input_field not in ('delta', 'rho'):
        raise ValueError("input_field must be 'delta' or 'rho'")

    # -- Output directories -------------------------------------------------
    for subdir in ('store_models', 'logs', 'output_data', 'losses'):
        os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)

    # -- Logging ------------------------------------------------------------
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    log_path = os.path.join(args.output_dir, 'logs', args.log_file)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s [%(name)s] [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logger = logging.getLogger(__name__)
    logger.info("Loss type: %s", args.loss_type)
    logger.info("Input field: %s, normalisation: %s", args.input_field,
                args.density_normalization)
    if args.seed is not None:
        logger.info("Random seed: %d", args.seed)

    # -- Collect data files -------------------------------------------------
    if not args.obs_dir or not args.true_dir:
        raise ValueError("obs_dir and true_dir must be provided")

    x_files = sorted(glob.glob(os.path.join(args.obs_dir, args.obs_basefile)))
    y_files = sorted(glob.glob(os.path.join(args.true_dir, args.true_basefile)))

    if args.min_mock_idx is not None and args.max_mock_idx is not None:
        x_files = x_files[args.min_mock_idx:args.max_mock_idx]
        y_files = y_files[args.min_mock_idx:args.max_mock_idx]
        logger.info("Using file indices %d to %d", args.min_mock_idx, args.max_mock_idx)

    if len(x_files) != len(y_files):
        raise ValueError(
            f"Mismatch in number of observed ({len(x_files)}) and "
            f"true ({len(y_files)}) files!"
        )
    logger.info("Found %d file pairs", len(x_files))

    # -- Mask handling ------------------------------------------------------
    single_mask = None
    if args.use_mask:
        mask_files = sorted(glob.glob(os.path.join(args.mask_dir, '*.npy'))) if args.mask_dir else []
        if not mask_files:
            raise ValueError("use_mask=True but no mask files found")
        if len(mask_files) > 1:
            raise NotImplementedError("Multiple per-sample masks not supported")
        single_mask = np.load(mask_files[0]).astype(np.float32)
        logger.info("Using single global mask: %s", mask_files[0])

    # For masked losses, mask is required
    if args.loss_type in ('masked_mse', 'masked_gradient') and single_mask is None:
        raise ValueError(
            f"loss_type='{args.loss_type}' requires --use_mask with a valid mask"
        )

    # -- Train/validation split ---------------------------------------------
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_files, y_files, test_size=0.2, shuffle=False,
    )
    logger.info("Training: %d, Validation: %d", len(x_train), len(x_valid))

    # -- Datasets -----------------------------------------------------------
    # Determine number of input channels
    if args.loss_type == 'mse':
        mask_channels = 2 if (args.use_mask and single_mask is not None) else 1
    else:
        mask_channels = 1  # masked losses use 1-channel input

    train_dataset = create_dataset(
        x_train, y_train, mask=single_mask,
        channels=mask_channels,
        batch_size=args.batch_size, shuffle=True,
        repeat=args.repeat_dataset, drop_remainder=args.drop_remainder,
        field_size=args.field_size, norm_val=args.density_normalization,
    )
    val_dataset = create_dataset(
        x_valid, y_valid, mask=single_mask,
        channels=mask_channels,
        batch_size=args.batch_size, shuffle=False,
        repeat=False, drop_remainder=False,
        field_size=args.field_size, norm_val=args.density_normalization,
    )

    # -- Build and compile model (multi-GPU) --------------------------------
    strategy = tf.distribute.MirroredStrategy()
    logger.info("Number of devices: %d", strategy.num_replicas_in_sync)
    keras.backend.clear_session()

    input_size = (args.field_size, args.field_size, args.field_size, mask_channels)

    with strategy.scope():
        # Build model
        if args.loss_type == 'mse':
            builder = UNet(
                base_filters=args.base_filters, min_size=args.min_size,
                dropout_layer=args.dropout, dropout_rate=args.dropout_rate,
                input_field=args.input_field,
                norm_val=args.density_normalization,
            )
            model = builder.prepare_model(input_size=input_size)
        else:
            inpainter = MaskedUNet3D(
                input_size=input_size,
                base_filters=args.base_filters, min_size=args.min_size,
                dropout_layer=args.dropout, dropout_rate=args.dropout_rate,
                use_mask=args.use_mask, input_field=args.input_field,
                norm_val=args.density_normalization,
            )
            model = inpainter.unet

        # Select loss
        if args.loss_type == 'mse':
            loss_fn = 'mean_squared_error'
        elif args.loss_type == 'masked_mse':
            single_mask_tf = tf.convert_to_tensor(single_mask[..., None], tf.float32)
            loss_fn = MaskedMSE(mask=single_mask_tf)
        elif args.loss_type == 'masked_gradient':
            single_mask_tf = tf.convert_to_tensor(single_mask[..., None], tf.float32)
            loss_fn = MaskedMSEWithGradient(
                mask=single_mask_tf, gradient_weight=args.gradient_weight,
            )

        # Optimiser
        opt_kwargs = dict(learning_rate=args.learning_rate, amsgrad=True)
        if args.global_clipnorm > 0:
            opt_kwargs['global_clipnorm'] = float(args.global_clipnorm)

        model.compile(
            optimizer=keras.optimizers.Adam(**opt_kwargs),
            loss=loss_fn,
            metrics=['mse'],
        )

    logger.info("Model built and compiled successfully")

    # -- Callbacks ----------------------------------------------------------
    checkpoint_cb = SaveEveryNEpoch(
        filepath=os.path.join(args.output_dir, 'store_models', 'model_{epoch:03d}.keras'),
        period=args.save_freq,
    )
    tensorboard_cb = keras.callbacks.TensorBoard(
        log_dir=os.path.join(args.output_dir, 'logs'), histogram_freq=1,
    )
    earlystop_cb = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=args.patience, verbose=2,
        restore_best_weights=True,
    )
    csv_logger = keras.callbacks.CSVLogger(
        os.path.join(args.output_dir, 'logs', 'history.csv'), append=False,
    )

    # -- Debug inspection ---------------------------------------------------
    if args.debug:
        logger.info("Debug: inspecting first 5 validation batches")
        for i, (x, y_true) in enumerate(val_dataset.take(5)):
            y_pred = model(x, training=False)
            logger.debug(
                "Batch %d -- input [%.4f, %.4f], target [%.4f, %.4f], "
                "pred [%.4f, %.4f], MSE %.6f",
                i,
                tf.reduce_min(x).numpy(), tf.reduce_max(x).numpy(),
                tf.reduce_min(y_true).numpy(), tf.reduce_max(y_true).numpy(),
                tf.reduce_min(y_pred).numpy(), tf.reduce_max(y_pred).numpy(),
                tf.reduce_mean(tf.square(y_true - y_pred)).numpy(),
            )

    # -- Training -----------------------------------------------------------
    steps_per_epoch = (
        int(np.ceil(len(x_train) / args.batch_size))
        if args.repeat_dataset else None
    )

    logger.info("Starting training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=[checkpoint_cb, tensorboard_cb, earlystop_cb, csv_logger],
        steps_per_epoch=steps_per_epoch,
        verbose=2,
    )
    logger.info("Training finished")

    # -- Save losses --------------------------------------------------------
    loss_dir = os.path.join(args.output_dir, 'losses')
    os.makedirs(loss_dir, exist_ok=True)
    np.save(os.path.join(loss_dir, 'loss.npy'), history.history['loss'])
    np.save(os.path.join(loss_dir, 'val_loss.npy'), history.history['val_loss'])

    # -- Evaluate on validation set -----------------------------------------
    logger.info("Evaluating on validation set...")
    val_dataset_eval = create_dataset(
        x_valid, y_valid, mask=single_mask,
        channels=mask_channels,
        batch_size=1, shuffle=False, repeat=False,
        field_size=args.field_size, norm_val=args.density_normalization,
    )

    predictions = model.predict(val_dataset_eval)
    for i, pred in enumerate(predictions):
        pred_field = pred[..., 0] * args.density_normalization
        if single_mask is not None:
            obs = np.load(x_valid[i]) * single_mask
            pred_field = obs * single_mask + pred_field * (1 - single_mask)
        np.save(
            os.path.join(args.output_dir, 'output_data', f'pred_field_{i:03d}.npy'),
            pred_field,
        )

    logger.info("Saved %d predicted fields", len(predictions))
    logger.info("Total execution time: %s", datetime.now() - start_time)


if __name__ == "__main__":
    main()
