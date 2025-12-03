#!/usr/bin/env python3
import os
import sys
import glob
import json
import logging
import numpy as np
import tensorflow as tf
from datetime import datetime

# -------------------------------
# GPU setup
# -------------------------------
physical_gpus = tf.config.list_physical_devices('GPU')
if physical_gpus:
    try:
        if len(physical_gpus) >= 2:
            tf.config.set_visible_devices(physical_gpus[2:4], 'GPU')
        else:
            tf.config.set_visible_devices(physical_gpus, 'GPU')
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, False)
    except Exception as e:
        print(f"Warning: could not configure GPUs: {e}")

from tensorflow import keras
from sklearn.model_selection import train_test_split

# -------------------------------
# Make project importable
# -------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from CosmoFiller.inpainting import MaskedInpaintingUNet, MaskedMSE
from CosmoFiller.datahandler import create_dataset


# ======================================================================
# MAIN
# ======================================================================
if __name__ == "__main__":
    # -------------------------------
    # Argument parser
    # -------------------------------
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate MaskedInpaintingUNet model on validation set")

    parser.add_argument('--param_file', type=str, help="JSON param file")
    parser.add_argument('--obs_dir', type=str, help='Directory with observed .npy fields')
    parser.add_argument('--true_dir', type=str, help='Directory with true .npy fields')
    parser.add_argument('--mask_dir', type=str, default=None, help='Directory with mask .npy file')
    parser.add_argument('--obs_basefile', type=str, default='*.npy')
    parser.add_argument('--true_basefile', type=str, default='*.npy')
    parser.add_argument('--use_mask', action='store_true')
    parser.add_argument('--input_field', type=str, choices=['rho', 'delta'], default='rho')
    parser.add_argument('--field_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--density_normalization', type=float, default=40.0)
    parser.add_argument('--base_filters', type=int, default=16)
    parser.add_argument('--min_size', type=int, default=4)
    parser.add_argument('--dropout', action='store_true')
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--global_clipnorm', type=float, default=1.0)
    parser.add_argument('--model_path', type=str, help='Path to saved .keras model')
    parser.add_argument('--output_dir', type=str, default='output_eval')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    # Load params from JSON
    if args.param_file:
        with open(args.param_file, 'r') as f:
            params = json.load(f)
        for key, value in params.items():
            if hasattr(args, key):
                setattr(args, key, value)

    if args.obs_dir is None or args.true_dir is None:
        raise ValueError("obs_dir and true_dir must be provided")

    # -------------------------------
    # Logging setup
    # -------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "output_data"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, "logs", "eval.log")),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    logger = logging.getLogger("eval")

    # -------------------------------
    # Collect data files
    # -------------------------------
    x_files = sorted(glob.glob(os.path.join(args.obs_dir, args.obs_basefile)))
    y_files = sorted(glob.glob(os.path.join(args.true_dir, args.true_basefile)))

    logger.info(f"Found {len(x_files)} observed fields")
    logger.info(f"Found {len(y_files)} true fields")

    # -------------------------------
    # Handle mask
    # -------------------------------
    mask_files = sorted(glob.glob(os.path.join(args.mask_dir, '*.npy'))) if args.use_mask else None

    single_mask = None
    if args.use_mask:
        if not mask_files:
            raise ValueError("use_mask=True but no mask file found")
        if len(mask_files) > 1:
            raise NotImplementedError("Multiple masks not supported")
        single_mask = np.load(mask_files[0]).astype(np.float32)
        logger.info(f"Loaded single mask: {mask_files[0]}")

    # -------------------------------
    # Train/validation split
    # -------------------------------
    _, x_valid, _, y_valid = train_test_split(
        x_files, y_files, test_size=0.2, shuffle=False
    )
    logger.info(f"Validation samples: {len(x_valid)}")

    # -------------------------------
    # Create validation dataset
    # -------------------------------
    val_dataset = create_dataset(
        x_valid,
        y_valid,
        single_mask=single_mask,
        batch_size=1,
        use_mask=args.use_mask,
        repeat=False,
        drop_remainder=False,
        field_size=args.field_size,
        norm_val=args.density_normalization
    )

    # -------------------------------
    # Load model
    # -------------------------------
    logger.info(f"Loading model from: {args.model_path}")

    custom_objects = {"MaskedMSE": MaskedMSE}

    model = keras.models.load_model(
        args.model_path,
        custom_objects=custom_objects,
        compile=True
    )

    logger.info("Model loaded successfully")

    # -------------------------------
    # Predict
    # -------------------------------
    logger.info("Predicting on validation set...")
    predictions = model.predict(val_dataset)

    # -------------------------------
    # Save predictions
    # -------------------------------
    for i, pred in enumerate(predictions):
        pred_field = pred[..., 0] * args.density_normalization

        if single_mask is not None:
            obs = np.load(x_valid[i])
            pred_field = obs * single_mask + pred_field * (1 - single_mask)

        out_file = os.path.join(
            args.output_dir, 'output_data', f'pred_field_{i:03d}.npy'
        )
        np.save(out_file, pred_field)

    logger.info("Evaluation completed successfully")
