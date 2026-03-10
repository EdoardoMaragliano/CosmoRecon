#!/usr/bin/env python3
"""Resume training a MaskedUNet3D from a saved checkpoint.

The initial epoch is extracted from the checkpoint filename (e.g.
``model_100.keras`` -> epoch 100).  If no checkpoint is specified, a fresh
model is built from scratch.
"""

import os
import sys
import re
import glob
import json
import logging

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# GPU setup -- uses all GPUs with memory growth enabled by default.
# For finer control, use the consolidated train.py script with --gpu_indices.
physical_gpus = tf.config.list_physical_devices('GPU')
if physical_gpus:
    try:
        tf.config.set_visible_devices(physical_gpus, 'GPU')
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, False)
    except Exception as e:
        print(f"Warning: could not configure GPUs: {e}")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from CosmoRecon.models import MaskedUNet3D, MaskedMSE
from CosmoRecon.checkpoints import SaveEveryNEpoch
from CosmoRecon.datahandler import create_dataset


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train or resume MaskedUNet3D")
    parser.add_argument('--param_file', type=str)
    parser.add_argument('--obs_dir', type=str)
    parser.add_argument('--true_dir', type=str)
    parser.add_argument('--mask_dir', type=str, default=None)
    parser.add_argument('--use_mask', action='store_true')
    parser.add_argument('--obs_basefile', type=str, default='*.npy')
    parser.add_argument('--true_basefile', type=str, default='*.npy')
    parser.add_argument('--input_field', type=str, choices=['rho', 'delta'], default='rho')
    parser.add_argument('--output_dir', type=str, default='output_products')
    parser.add_argument('--field_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--drop_remainder', action='store_true')
    parser.add_argument('--repeat_dataset', action='store_true')
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--base_filters', type=int, default=16)
    parser.add_argument('--min_size', type=int, default=4)
    parser.add_argument('--dropout', action='store_true')
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--global_clipnorm', type=float, default=1.0)
    parser.add_argument('--density_normalization', type=float, default=40.0)
    parser.add_argument('--patience', type=int, default=80)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to a .keras checkpoint to resume from')
    args = parser.parse_args()

    if args.param_file:
        with open(args.param_file, 'r') as f:
            params = json.load(f)
        for key, value in params.items():
            if hasattr(args, key):
                setattr(args, key, value)
            else:
                logging.warning("Unknown JSON key '%s' ignored.", key)

    if not args.obs_dir or not args.true_dir:
        raise ValueError("obs_dir and true_dir must be provided.")

    # Output directories
    for subdir in ('store_models', 'output_data', 'logs'):
        os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)

    # Logging
    log_path = os.path.join(args.output_dir, 'logs', 'train.log')
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("trainer")

    # Collect files
    x_files = sorted(glob.glob(os.path.join(args.obs_dir, args.obs_basefile)))
    y_files = sorted(glob.glob(os.path.join(args.true_dir, args.true_basefile)))
    logger.info("Found %d X, %d Y files", len(x_files), len(y_files))

    # Mask
    single_mask = None
    if args.use_mask:
        mask_files = sorted(glob.glob(os.path.join(args.mask_dir, '*.npy'))) if args.mask_dir else []
        if not mask_files:
            raise ValueError("use_mask=True but no mask files found.")
        if len(mask_files) > 1:
            raise NotImplementedError("Multiple masks not supported.")
        single_mask = np.load(mask_files[0]).astype(np.float32)
        logger.info("Using single global mask.")

    # Split
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_files, y_files, test_size=0.2, shuffle=False,
    )
    logger.info("Train: %d, Validation: %d", len(x_train), len(x_valid))

    # Build or load model
    if args.resume_from:
        logger.info("Resuming from checkpoint: %s", args.resume_from)
        model = keras.models.load_model(
            args.resume_from, custom_objects={"MaskedMSE": MaskedMSE},
        )
        # Re-inject mask into MaskedMSE loss after deserialization
        # (mask is serialized as None since tf.Variable is not JSON-safe)
        if single_mask is not None and hasattr(model, 'loss'):
            single_mask_tf = tf.convert_to_tensor(single_mask[..., None], tf.float32)
            if isinstance(model.loss, MaskedMSE):
                model.loss.set_mask(single_mask_tf)
                logger.info("Re-injected mask into MaskedMSE loss.")
            elif hasattr(model.loss, 'set_mask'):
                model.loss.set_mask(single_mask_tf)
                logger.info("Re-injected mask into loss function.")
        match = re.search(r"model_(\d+)\.keras", args.resume_from)
        initial_epoch = int(match.group(1)) if match else 0
        if not match:
            logger.warning("Could not extract epoch from filename; starting from 0")
    else:
        logger.info("Building new model...")
        inpainter = MaskedUNet3D(
            input_size=(args.field_size, args.field_size, args.field_size, 1),
            base_filters=args.base_filters, min_size=args.min_size,
            dropout_layer=args.dropout, dropout_rate=args.dropout_rate,
            use_mask=args.use_mask, input_field=args.input_field,
            norm_val=args.density_normalization,
        )
        model = inpainter.unet

        opt_kwargs = dict(learning_rate=args.learning_rate)
        if args.global_clipnorm > 0:
            opt_kwargs['clipnorm'] = float(args.global_clipnorm)
        optimizer = keras.optimizers.Adam(**opt_kwargs)

        model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
        initial_epoch = 0

    # Datasets
    train_dataset = create_dataset(
        x_train, y_train, mask=single_mask,
        batch_size=args.batch_size, repeat=args.repeat_dataset,
        drop_remainder=args.drop_remainder,
        field_size=args.field_size, norm_val=args.density_normalization,
    )
    val_dataset = create_dataset(
        x_valid, y_valid, mask=single_mask,
        batch_size=args.batch_size, repeat=False, drop_remainder=False,
        field_size=args.field_size, norm_val=args.density_normalization,
    )

    # Callbacks
    checkpoint_cb = SaveEveryNEpoch(
        filepath=os.path.join(args.output_dir, 'store_models', 'model_{epoch:03d}.keras'),
        period=args.save_freq,
    )
    earlystop_cb = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=args.patience, restore_best_weights=True,
    )
    csv_logger = keras.callbacks.CSVLogger(
        os.path.join(args.output_dir, 'history.csv'),
        append=(args.resume_from is not None),
    )

    # Train
    logger.info("Starting training (initial_epoch=%d)...", initial_epoch)
    model.fit(
        train_dataset, validation_data=val_dataset,
        epochs=args.epochs, initial_epoch=initial_epoch,
        callbacks=[checkpoint_cb, earlystop_cb, csv_logger],
        verbose=2,
    )

    # Evaluate
    logger.info("Predicting on validation set...")
    val_dataset_eval = create_dataset(
        x_valid, y_valid, mask=single_mask,
        batch_size=1, repeat=False, drop_remainder=False,
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

    logger.info("Training + evaluation completed.")


if __name__ == "__main__":
    main()
