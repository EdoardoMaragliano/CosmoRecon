"""
Extracts training parameters from a JSON file or command-line arguments,
builds or loads a 3D MaskedInpaintingUNet model, and resumes training from a saved checkpoint.
Saves model checkpoints, training logs, and evaluation outputs.

initial epoch is determined from the loaded model filename.
training continues from that epoch up to the specified total number of epochs.
"""


#!/usr/bin/env python3
import os
import sys
import glob
import json
import logging
from datetime import datetime
import numpy as np
import tensorflow as tf

# ------------------------------------------------
# GPU CONFIG
# ------------------------------------------------
physical_gpus = tf.config.list_physical_devices('GPU')
if physical_gpus:
    try:
        tf.config.set_visible_devices(physical_gpus, 'GPU')
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"Warning: could not configure GPUs: {e}")

from tensorflow import keras
from sklearn.model_selection import train_test_split

# ------------------------------------------------
# Insert project root into PYTHONPATH
# ------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Imports from your project
from CosmoFiller.inpainting import MaskedInpaintingUNet, MaskedMSE
from CosmoFiller.checkpoints import SaveEveryNEpoch
from CosmoFiller.datahandler import create_dataset

# ------------------------------------------------
# MAIN
# ------------------------------------------------
if __name__ == "__main__":

    # ================================================================
    # ARGUMENT PARSER
    # ================================================================
    import argparse
    parser = argparse.ArgumentParser(description="Train or resume training of 3D MaskedInpaintingUNet")

    parser.add_argument('--param_file', type=str, help="JSON param file with training settings")

    # Required inputs
    parser.add_argument('--obs_dir', type=str, help='Directory with observed .npy fields')
    parser.add_argument('--true_dir', type=str, help='Directory with true .npy fields')

    # Optional mask
    parser.add_argument('--mask_dir', type=str, default=None, help='Directory with mask .npy files (optional)')
    parser.add_argument('--use_mask', action='store_true', help='Enable mask usage')

    # File patterns
    parser.add_argument('--obs_basefile', type=str, default='*.npy')
    parser.add_argument('--true_basefile', type=str, default='*.npy')

    # Model + training parameters
    parser.add_argument('--input_field', type=str, choices=['rho', 'delta'], default='rho')
    parser.add_argument('--output_dir', type=str, default='output_products')
    parser.add_argument('--field_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--drop_remainder', action='store_true')
    parser.add_argument('--repeat_dataset', action='store_true')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--base_filters', type=int, default=16)
    parser.add_argument('--min_size', type=int, default=4)
    parser.add_argument('--dropout', action='store_true')
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--global_clipnorm', type=float, default=1.0)
    parser.add_argument('--density_normalization', type=float, default=40.0)
    parser.add_argument('--patience', type=int, default=80)
    parser.add_argument('--debug', action='store_true')

    # Manual resume option
    parser.add_argument('--resume_from', type=str, default=None,
                        help="Path to a saved .keras checkpoint to resume training from")

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Load JSON config (overrides CLI args)
    # ------------------------------------------------------------
    if args.param_file:
        with open(args.param_file, 'r') as f:
            params = json.load(f)
        for key, value in params.items():
            if hasattr(args, key):
                setattr(args, key, value)

    if args.obs_dir is None or args.true_dir is None:
        raise ValueError("obs_dir and true_dir must be provided.")

    # ================================================================
    # OUTPUT DIRECTORIES
    # ================================================================
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'store_models'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'output_data'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)

    # ================================================================
    # LOGGING
    # ================================================================
    log_path = os.path.join(args.output_dir, 'logs', 'train.log')
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("trainer")
    logger.info("Starting script")

    # ================================================================
    # COLLECT FILES
    # ================================================================
    x_files = sorted(glob.glob(os.path.join(args.obs_dir, args.obs_basefile)))
    y_files = sorted(glob.glob(os.path.join(args.true_dir, args.true_basefile)))

    mask_files = None
    if args.use_mask and args.mask_dir:
        mask_files = sorted(glob.glob(os.path.join(args.mask_dir, "*.npy")))

    logger.info(f"Found {len(x_files)} X files")
    logger.info(f"Found {len(y_files)} Y files")

    # ================================================================
    # SINGLE MASK HANDLING
    # ================================================================
    single_mask = None
    if args.use_mask:
        if mask_files is None or len(mask_files) == 0:
            raise ValueError("use_mask=True but no mask files provided.")
        elif len(mask_files) == 1:
            single_mask = np.load(mask_files[0]).astype(np.float32)
            logger.info("Using single global mask.")
        else:
            raise NotImplementedError("Multiple mask files are not supported.")

    # ================================================================
    # TRAIN/VAL SPLIT
    # ================================================================
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_files, y_files, test_size=0.2, shuffle=False
    )

    logger.info(f"Training samples: {len(x_train)}")
    logger.info(f"Validation samples: {len(x_valid)}")

    # ================================================================
    # BUILD/LOAD MODEL
    # ================================================================
    if args.resume_from is not None:
        # --------------------------------------------------------
        # LOAD SAVED MODEL (FULL OPTIMIZER + EPOCH STATE)
        # --------------------------------------------------------
        logger.info(f"Resuming training from checkpoint: {args.resume_from}")
        model = keras.models.load_model(
            args.resume_from,
            custom_objects={"MaskedMSE": MaskedMSE}
        )

        # extract epoch number
        import re
        m = re.search(r"model_(\d+)\.keras", args.resume_from)
        if m:
            initial_epoch = int(m.group(1))
        else:
            initial_epoch = 0
            logger.warning("Could not extract epoch number — starting from epoch=0.")

    else:
        # --------------------------------------------------------
        # NEW MODEL
        # --------------------------------------------------------
        logger.info("Building new model...")
        inpainter = MaskedInpaintingUNet(
            input_size=args.field_size,
            base_filters=args.base_filters,
            min_size=args.min_size,
            dropout_layer=args.dropout,
            dropout_rate=args.dropout_rate,
            use_mask=args.use_mask,
            input_field=args.input_field,
            norm_val=args.density_normalization
        )
        model = inpainter.unet

        # optimizer
        if args.global_clipnorm > 0:
            optimizer = keras.optimizers.Adam(
                learning_rate=args.learning_rate,
                clipnorm=float(args.global_clipnorm)
            )
        else:
            optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)

        model.compile(
            optimizer=optimizer,
            loss=MaskedMSE(),
            metrics=[MaskedMSE()]
        )

        initial_epoch = 0

    # ================================================================
    # DATASETS
    # ================================================================
    train_dataset = create_dataset(
        x_train, y_train,
        single_mask=single_mask,
        batch_size=args.batch_size,
        use_mask=args.use_mask,
        repeat=args.repeat_dataset,
        drop_remainder=args.drop_remainder,
        field_size=args.field_size,
        norm_val=args.density_normalization
    )

    val_dataset = create_dataset(
        x_valid, y_valid,
        single_mask=single_mask,
        batch_size=args.batch_size,
        use_mask=args.use_mask,
        repeat=False,
        drop_remainder=False,
        field_size=args.field_size,
        norm_val=args.density_normalization
    )

    # ================================================================
    # CALLBACKS
    # ================================================================
    checkpoint_cb = SaveEveryNEpoch(
        filepath=os.path.join(args.output_dir, 'store_models', 'model_{epoch:03d}.keras'),
        period=100
    )
    earlystop_cb = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=args.patience,
        restore_best_weights=True
    )
    csv_logger = keras.callbacks.CSVLogger(
        os.path.join(args.output_dir, 'history.csv'),
        append=(args.resume_from is not None)
    )

    # ================================================================
    # TRAINING
    # ================================================================
    logger.info("Starting training...")

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        initial_epoch=initial_epoch,
        callbacks=[checkpoint_cb, earlystop_cb, csv_logger],
        verbose=2
    )

    # ================================================================
    # EVALUATION ON VAL SET
    # ================================================================
    logger.info("Predicting on validation set...")

    val_dataset_eval = create_dataset(
        x_valid, y_valid,
        single_mask=single_mask,
        batch_size=1,
        use_mask=args.use_mask,
        repeat=False,
        drop_remainder=False,
        field_size=args.field_size,
        norm_val=args.density_normalization
    )

    predictions = model.predict(val_dataset_eval)

    for i, pred in enumerate(predictions):
        pred_field = pred[..., 0] * args.density_normalization

        if single_mask is not None:
            obs_sample = np.load(x_valid[i]) * single_mask
            pred_field = obs_sample * single_mask + pred_field * (1 - single_mask)

        np.save(
            os.path.join(args.output_dir, 'output_data', f'pred_field_{i:03d}.npy'),
            pred_field
        )

    logger.info("Training + evaluation completed successfully.")
