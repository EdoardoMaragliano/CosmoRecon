#!/usr/bin/env python3
"""Evaluate a saved model on the validation split.

Loads a trained Keras model, runs inference on the held-out validation set,
and saves the de-normalised predicted fields to disk.
"""

import os
import sys
import glob
import json
import logging

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# GPU setup
physical_gpus = tf.config.list_physical_devices('GPU')
if physical_gpus:
    try:
        tf.config.set_visible_devices(physical_gpus[:2], 'GPU')
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, False)
    except Exception as e:
        print(f"Warning: could not configure GPUs: {e}")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from CosmoRecon.models import MaskedMSE
from CosmoRecon.datahandler import create_dataset


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument('--param_file', type=str)
    parser.add_argument('--obs_dir', type=str, help='Observed .npy directory')
    parser.add_argument('--true_dir', type=str, help='True .npy directory')
    parser.add_argument('--mask_dir', type=str, default=None)
    parser.add_argument('--obs_basefile', type=str, default='*.npy')
    parser.add_argument('--true_basefile', type=str, default='*.npy')
    parser.add_argument('--use_mask', action='store_true')
    parser.add_argument('--input_field', type=str, choices=['rho', 'delta'], default='rho')
    parser.add_argument('--field_size', type=int, default=128)
    parser.add_argument('--density_normalization', type=float, default=40.0)
    parser.add_argument('--model_path', type=str, help='Path to saved .keras model')
    parser.add_argument('--output_dir', type=str, default='output_eval')
    parser.add_argument('--debug', action='store_true')
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
        raise ValueError("obs_dir and true_dir must be provided")

    # Output directories
    for subdir in ('output_data', 'logs'):
        os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)

    # Logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'logs', 'eval.log')),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    logger = logging.getLogger("eval")

    # Collect files
    x_files = sorted(glob.glob(os.path.join(args.obs_dir, args.obs_basefile)))
    y_files = sorted(glob.glob(os.path.join(args.true_dir, args.true_basefile)))
    logger.info("Found %d observed, %d true fields", len(x_files), len(y_files))

    # Mask
    single_mask = None
    if args.use_mask:
        mask_files = sorted(glob.glob(os.path.join(args.mask_dir, '*.npy')))
        if not mask_files:
            raise ValueError("use_mask=True but no mask files found")
        if len(mask_files) > 1:
            raise NotImplementedError("Multiple masks not supported")
        single_mask = np.load(mask_files[0]).astype(np.float32)
        logger.info("Loaded single mask: %s", mask_files[0])

    # Reproduce the same train/val split
    _, x_valid, _, y_valid = train_test_split(
        x_files, y_files, test_size=0.2, shuffle=False,
    )
    logger.info("Validation samples: %d", len(x_valid))

    # Dataset
    val_dataset = create_dataset(
        x_valid, y_valid, mask=single_mask,
        batch_size=1, repeat=False, drop_remainder=False,
        field_size=args.field_size, norm_val=args.density_normalization,
    )

    # Load model
    logger.info("Loading model from: %s", args.model_path)
    model = keras.models.load_model(
        args.model_path,
        custom_objects={"MaskedMSE": MaskedMSE},
        compile=True,
    )
    logger.info("Model loaded")

    # Predict and save
    logger.info("Predicting on validation set...")
    predictions = model.predict(val_dataset)

    for i, pred in enumerate(predictions):
        pred_field = pred[..., 0] * args.density_normalization
        if single_mask is not None:
            obs = np.load(x_valid[i])
            pred_field = obs * single_mask + pred_field * (1 - single_mask)

        np.save(
            os.path.join(args.output_dir, 'output_data', f'pred_field_{i:03d}.npy'),
            pred_field,
        )

    logger.info("Evaluation completed (%d fields saved)", len(predictions))


if __name__ == "__main__":
    main()
