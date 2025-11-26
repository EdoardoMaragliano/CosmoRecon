#!/usr/bin/env python3
import os
import sys
import glob
import json
import logging
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Inserisci la root del progetto in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import your modules
from CosmoFiller.inpainting import MaskedInpaintingUNet
from CosmoFiller.checkpoints import SaveEveryNEpoch
from CosmoFiller.datahandler import create_dataset

# -------------------------------
# Main script
# -------------------------------
if __name__ == "__main__":
    # -------------------------------
    # Argument parser
    # -------------------------------
    import argparse
    parser = argparse.ArgumentParser(description="Train 3D MaskedInpaintingUNet with MirroredStrategy")
    parser.add_argument('--param_file', type=str, help="JSON param file with training settings")

    # -------------------------------
    # Default parameters
    # -------------------------------
    parser.add_argument('--obs_dir', type=str, help='Directory with observed .npy fields')
    parser.add_argument('--true_dir', type=str, help='Directory with true .npy fields')
    parser.add_argument('--mask_dir', type=str, default=None, help='Directory with mask .npy files (optional)')
    parser.add_argument('--obs_basefile', type=str, default='*.npy', help='Glob pattern for observed fields')
    parser.add_argument('--true_basefile', type=str, default='*.npy', help='Glob pattern for true fields')
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
        for key, value in params.items():
            if hasattr(args, key):
                setattr(args, key, value)

    # Dopo aver fatto override dal JSON
    if args.obs_dir is None or args.true_dir is None:
        raise ValueError("obs_dir and true_dir must be provided either via CLI or JSON")

    # -------------------------------
    # Create output directories
    # -------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'store_models'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'output_data'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)

    # -------------------------------
    # Setup logging
    # -------------------------------
    # Clear root handlers
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    log_path = os.path.join(args.output_dir, 'logs', args.log_file)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s [%(name)s] [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    logger = logging.getLogger("training")
    logger.info("Starting training script")

    # -------------------------------
    # Collect files
    # -------------------------------
    x_files = sorted(glob.glob(os.path.join(args.obs_dir, args.obs_basefile)))
    y_files = sorted(glob.glob(os.path.join(args.true_dir, args.true_basefile)))
    mask_files = sorted(glob.glob(os.path.join(args.mask_dir, '*.npy'))) if args.use_mask else None

    logger.info(f"x_files are: {x_files[0]}")
    logger.info(f"y_files are: {y_files[0]}")
    if mask_files:
        logger.info(f"mask_files are: {mask_files[0]}")
    logger.info(f"Collected {len(x_files)} observed files, {len(y_files)} true files, and {len(mask_files) if mask_files else 0} mask files")

    if args.use_mask and mask_files and len(mask_files) == 1:
        single_mask = np.load(mask_files[0]).astype(np.float32)
        mask_files = None
        logger.info(f"Using single mask for all samples: {mask_files[0]}")
    else:
        single_mask = None

    # -------------------------------
    # Train/validation split
    # -------------------------------
    if args.use_mask and mask_files is not None:
        x_train, x_valid, y_train, y_valid, mask_train, mask_valid = train_test_split(
            x_files, y_files, mask_files, test_size=0.2, shuffle=False)
    else:
        x_train, x_valid, y_train, y_valid = train_test_split(x_files, y_files, test_size=0.2, shuffle=False)
        mask_train = mask_valid = None

    logger.info(f"Train samples: {len(x_train)}, Validation samples: {len(x_valid)}")

    # -------------------------------
    # Create datasets
    # -------------------------------
    train_dataset = create_dataset(
        x_train, y_train,
        mask_files=mask_train, single_mask=single_mask,
        batch_size=args.batch_size, use_mask=args.use_mask,
        repeat=True, drop_remainder=True,
        field_size=args.field_size,
        norm_val=args.density_normalization
    )

    val_dataset = create_dataset(
        x_valid, y_valid,
        mask_files=mask_valid, single_mask=single_mask,
        batch_size=args.batch_size, use_mask=args.use_mask,
        repeat=False, drop_remainder=False,
        field_size=args.field_size,
        norm_val=args.density_normalization
    )

    # -------------------------------
    # Multi-GPU strategy
    # -------------------------------
    strategy = tf.distribute.MirroredStrategy()
    logger.info(f"Number of devices: {strategy.num_replicas_in_sync}")

    input_channels = 2 if args.use_mask else 1
    input_size = (args.field_size, args.field_size, args.field_size, input_channels)

    with strategy.scope():
        model = MaskedInpaintingUNet(
            input_size=input_size,
            base_filters=args.base_filters,
            min_size=args.min_size,
            dropout_layer=args.dropout,
            dropout_rate=args.dropout_rate,
            use_mask=args.use_mask,
            input_field=args.input_field,
            norm_val=args.density_normalization
        )

        optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate, )
        if args.global_clipnorm and args.global_clipnorm > 0:
            optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate, clipnorm=float(args.global_clipnorm))

        model.compile(
            optimizer=optimizer,
            loss=None,  # masked loss è gestita internamente
            metrics=['mean_squared_error']
        )

    # -------------------------------
    # Callbacks
    # -------------------------------
    checkpoint_cb = SaveEveryNEpoch(
        filepath=os.path.join(args.output_dir, 'store_models', 'model_{epoch:03d}.keras'),
        period=10
    )
    earlystop_cb = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=20, restore_best_weights=True
    )
    csv_logger = keras.callbacks.CSVLogger(
        os.path.join(args.output_dir, 'history.csv'), append=False
    )

    # -------------------------------
    # Training
    # -------------------------------
    logger.info("Starting training...")
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=[checkpoint_cb, earlystop_cb, csv_logger],
        verbose=2
    )

    # -------------------------------
    # Evaluation: save predicted fields
    # -------------------------------
    val_dataset_eval = create_dataset(
        x_valid, y_valid,
        mask_files=mask_valid, single_mask=single_mask,
        batch_size=1, use_mask=args.use_mask,
        repeat=False, drop_remainder=False,
        field_size=args.field_size,
        norm_val=args.density_normalization
    )

    logger.info("Predicting on validation set...")
    predictions = model.predict(val_dataset_eval)
    for i, pred in enumerate(predictions):
        pred_field = pred[..., 0] * args.density_normalization

        if single_mask is not None:
            obs_sample = np.load(x_valid[i]) * single_mask
    
            # join the observed field + predicted field in masked regions
            pred_field = np.where(single_mask[..., 0]==1,obs_sample, pred_field)

        np.save(os.path.join(args.output_dir, 'output_data', f'pred_field_{i:03d}.npy'), pred_field)

    logger.info("Training and evaluation completed successfully")
