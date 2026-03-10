#!/usr/bin/env python3
"""Hyper-parameter optimisation with Optuna.

Searches over batch size and learning rate using short training runs and
reports the best configuration based on validation MSE.
"""

import argparse
import glob
import json
import logging
import os
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import optuna

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from CosmoRecon.models import UNet
from CosmoRecon.datahandler import create_dataset


def train_trial(x_train, y_train, x_valid, y_valid, single_mask, mask_channels,
                field_size, input_field, density_normalization,
                batch_size, learning_rate, epochs_per_trial):
    """Run a short training and return the final validation loss."""
    train_ds = create_dataset(
        x_train, y_train, mask=single_mask, channels=mask_channels,
        batch_size=batch_size, shuffle=True,
        field_size=field_size, norm_val=density_normalization,
    )
    val_ds = create_dataset(
        x_valid, y_valid, mask=single_mask, channels=mask_channels,
        batch_size=batch_size, shuffle=False,
        field_size=field_size, norm_val=density_normalization,
    )

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        builder = UNet(input_field=input_field, norm_val=density_normalization)
        model = builder.prepare_model(
            input_size=(field_size, field_size, field_size, mask_channels),
        )
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=learning_rate, amsgrad=True, global_clipnorm=1e-2,
            ),
            loss='mean_squared_error',
            metrics=['mean_squared_error'],
        )

    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=epochs_per_trial, verbose=0,
    )
    return history.history['val_loss'][-1]


def main():
    parser = argparse.ArgumentParser(description="Optuna hyper-parameter search")
    parser.add_argument('--obs_dir', type=str, default='path_to_obs')
    parser.add_argument('--true_dir', type=str, default='path_to_true')
    parser.add_argument('--mask_dir', type=str, default=None,
                        help='Directory with mask .npy files (None = no mask)')
    parser.add_argument('--input_field', type=str, default='rho',
                        choices=['rho', 'delta'])
    parser.add_argument('--field_size', type=int, default=128)
    parser.add_argument('--epochs_per_trial', type=int, default=5)
    parser.add_argument('--n_trials', type=int, default=20)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--param_file', type=str, default=None,
                        help='JSON parameter file for overrides')
    args = parser.parse_args()

    if args.param_file:
        with open(args.param_file, 'r') as f:
            params = json.load(f)
        for key, value in params.items():
            if hasattr(args, key):
                setattr(args, key, value)
            else:
                logging.warning("Unknown JSON key '%s' ignored.", key)

    output_dir = args.output_dir or os.path.join(
        "optuna_results", datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    os.makedirs(output_dir, exist_ok=True)

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(output_dir, "optuna.log")),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting hyper-parameter search")
    logger.info("OBS_DIR=%s, TRUE_DIR=%s, MASK_DIR=%s",
                args.obs_dir, args.true_dir, args.mask_dir)

    # Data
    x_files = sorted(glob.glob(os.path.join(args.obs_dir, '*.npy')))
    y_files = sorted(glob.glob(os.path.join(args.true_dir, '*.npy')))
    if len(x_files) != len(y_files):
        raise ValueError(
            f"Mismatch between obs ({len(x_files)}) and true ({len(y_files)}) files!"
        )

    density_normalization = 40.0 if args.input_field == "rho" else 20.0

    x_train, x_valid, y_train, y_valid = train_test_split(
        x_files, y_files, test_size=args.test_size, shuffle=True,
    )

    # Mask handling
    use_mask = args.mask_dir is not None
    single_mask = None
    if use_mask:
        mask_files = sorted(glob.glob(os.path.join(args.mask_dir, '*.npy')))
        if len(mask_files) == 1:
            single_mask = np.load(mask_files[0]).astype(np.float32)

    mask_channels = 2 if (use_mask and single_mask is not None) else 1

    def objective(trial):
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        val_loss = train_trial(
            x_train, y_train, x_valid, y_valid,
            single_mask, mask_channels,
            args.field_size, args.input_field, density_normalization,
            batch_size, learning_rate, args.epochs_per_trial,
        )
        logger.info(
            "Trial %d: batch_size=%d, lr=%.2e, val_loss=%.6f",
            trial.number, batch_size, learning_rate, val_loss,
        )
        return val_loss

    # Run study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.n_trials)

    best = study.best_trial
    logger.info("Best trial: batch_size=%d, lr=%.2e, val_loss=%.6f",
                best.params['batch_size'], best.params['learning_rate'], best.value)

    # Save results
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'best_trial.json'), 'w') as f:
        json.dump({
            'learning_rate': best.params['learning_rate'],
            'batch_size': best.params['batch_size'],
            'val_loss': best.value,
        }, f, indent=2)

    with open(os.path.join(results_dir, 'study_summary.txt'), 'w') as f:
        f.write(f"Number of trials: {len(study.trials)}\n")
        f.write(f"Best trial:\n")
        f.write(f"  Learning rate: {best.params['learning_rate']}\n")
        f.write(f"  Batch size: {best.params['batch_size']}\n")
        f.write(f"  Validation loss: {best.value}\n")

    logger.info("Results saved to %s", results_dir)


if __name__ == "__main__":
    main()
