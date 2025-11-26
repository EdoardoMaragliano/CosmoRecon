#!/usr/bin/env python3
import os

import sys
# Inserisci la root del progetto in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


import glob
import argparse
import logging

import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from datetime import datetime
from CosmoFiller.inpainting import UNet
from CosmoFiller.datahandler import create_dataset
from CosmoFiller.utils import setup_logger

start_time = datetime.now()  # inizio timer

# ============================
# Command-line arguments
# ============================
parser = argparse.ArgumentParser(description='Train 3D U-Net for inpainting.')
parser.add_argument('--param_file', type=str, help='Path to JSON parameter file')
parser.add_argument('--obs_dir', type=str, help='Directory with observed .npy fields')
parser.add_argument('--true_dir', type=str, help='Directory with true .npy fields')
parser.add_argument('--mask_dir', type=str, help='Directory with mask .npy files')
parser.add_argument('--obs_basefile', type=str, help='Base filename for observed .npy fields')
parser.add_argument('--true_basefile', type=str, help='Base filename for true .npy fields')
parser.add_argument('--input_field', type=str, default='rho', help='Type of input field: delta or rho')
parser.add_argument('--output_dir', type=str, default='output_products', help='Directory to store output fields')
parser.add_argument('--use_mask', type=bool, default=False, help='True = use mask, False = do not use mask')
parser.add_argument('--field_size', type=int, default=128, help='Size of input fields (NxNxN)')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--repeat_dataset', type=bool, default=False, help='Repeat dataset indefinitely for continuous training')
parser.add_argument('--drop_remainder', type=bool, default=False, help='Drop remainder in batching to ensure consistent batch sizes')
parser.add_argument('--save_freq', type=int, default=10, help='Frequency (in epochs) to save the model')
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--log_file', type=str, default='training.log', help='File name (not path) to store logs')
parser.add_argument('--debug', action='store_true', help='Enable debug mode with verbose logging')
parser.add_argument('--density_normalization', type=float, default=40.0, help='Normalization factor for density fields')
parser.add_argument('--min_mock_idx', type=int, default=None, help='Minimum mock index to use (for subsetting)')
parser.add_argument('--max_mock_idx', type=int, default=None, help='Maximum mock index to use (for subsetting)')


args = parser.parse_args()

# Ensure output directories exist
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'store_models'), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'output_data'), exist_ok=True)

# ============================
# Load parameters from JSON
# ============================
if args.param_file:
    with open(args.param_file, 'r') as f:
        params = json.load(f)
    for key, value in params.items():
        if hasattr(args, key):
            setattr(args, key, value)

use_mask = bool(args.use_mask)

if args.input_field not in ['delta', 'rho']:
    raise ValueError("input_field must be 'delta' or 'rho'")

density_normalization = args.density_normalization


# ============================
# Setup logging
# ============================
# Clear existing handlers for the root logger
for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)

log_path = os.path.join(args.output_dir, 'logs', args.log_file)
os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    

# Main logger
logging.basicConfig(
    level=logging.INFO if not args.debug else logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)

if args.input_field == 'delta':
    logging.info("Training to reconstruct delta fields (shifted ReLU output)")
    logging.info(f"Using density normalization: {density_normalization}")
elif args.input_field == 'rho':
    logging.info("Training to reconstruct rho fields (standard output)")
    logging.info(f"Using density normalization: {density_normalization}")

# ============================
# Collect files
# ============================
x_files = sorted(glob.glob(os.path.join(args.obs_dir,
             args.obs_basefile)))
y_files = sorted(glob.glob(os.path.join(args.true_dir, 
            args.true_basefile)))
mask_files = sorted(glob.glob(os.path.join(args.mask_dir, '*.npy')))

# select indexes from args if needed (not implemented here)
if args.min_mock_idx is not None and args.max_mock_idx is not None:
    x_files = x_files[args.min_mock_idx:args.max_mock_idx]
    y_files = y_files[args.min_mock_idx:args.max_mock_idx]
    if use_mask:
        mask_files = mask_files[args.min_mock_idx:args.max_mock_idx]

        logging.info(f"Using files from index {args.min_mock_idx} to {args.max_mock_idx}")
        logging.info(f"Number of x_files: {len(x_files)}, y_files: {len(y_files)}, mask_files: {len(mask_files)}")


assert len(x_files) == len(y_files), "Mismatch in number of observed and true files!"

# Single mask check
if use_mask and mask_files:
    if len(mask_files) == 1:
        # if there is only one mask file, use it for all samples
        single_mask = np.load(mask_files[0]).astype(np.float32)
        use_single_mask = True
        logging.info("Using a single mask for all samples")
    else:
        # Multiple mask files, one for each sample
        single_mask = None
        use_single_mask = False
        assert len(mask_files) == len(x_files), "Mismatch in number of mask files!"
        logging.info(f"Using individual masks for each sample ({len(mask_files)} files)")
else:
    # No mask will be used
    single_mask = None
    use_single_mask = False
    use_mask = False
    logging.info("No mask will be used")

logging.info(f"Found {len(x_files)} files for training")

# ============================
# Split dataset
# ============================
if use_mask and use_single_mask:
    # Use the same single mask for all samples -> no need to split mask files
    x_train, x_valid, y_train, y_valid = train_test_split(x_files, y_files, test_size=0.2, shuffle=False)
    mask_train = mask_valid = None
    logging.info("Reading mask from: %s", mask_files[0])
    single_mask = np.load(mask_files[0]).astype(np.float32)
    logging.info("Single mask shape: %s", single_mask.shape)
elif use_mask and not use_single_mask:
    # Use individual masks for each sample -> split mask files as well
    x_train, x_valid, y_train, y_valid, mask_train, mask_valid = train_test_split(
        x_files, y_files, mask_files, test_size=0.2, shuffle=False)
    for m in mask_train[:3]:
        logging.debug("Training mask example: %s", m)
    for m in mask_valid[:3]:
        logging.debug("Validation mask example: %s", m)
else:
    # No masks used
    x_train, x_valid, y_train, y_valid = train_test_split(x_files, y_files, test_size=0.2, shuffle=False)
    mask_train = mask_valid = None
    logging.info("No mask will be used")

for x in x_train[:3]:
    logging.info("Training obs example: %s", x)
for y in y_train[:3]:
    logging.info("Training true example: %s", y)
for x in x_valid[:3]:
    logging.info("Validation obs example: %s", x)
for y in y_valid[:3]:
    logging.info("Validation true example: %s", y)

logging.info(f"Training samples: {len(x_train)}, Validation samples: {len(x_valid)}")

############################
## create datasets
###########################
train_dataset = create_dataset(
    x_train, y_train,
    mask_files = mask_train if mask_train is not None else None,
    single_mask=single_mask if use_single_mask else None,
    batch_size = args.batch_size,
    use_mask = use_mask,
    shuffle = True,
    repeat = args.repeat_dataset,
    drop_remainder=False,
    field_size=args.field_size,
    norm_val=density_normalization
    )

logging.info('train dataset has shape: %s', str(train_dataset))

val_dataset = create_dataset(
    x_valid, y_valid,
    mask_files = mask_valid if mask_valid is not None else None,
    single_mask = single_mask if use_single_mask else None,
    batch_size = args.batch_size,
    shuffle = False,                  # No shuffle for validation
    use_mask = use_mask,
    repeat = False,                    # No repeat for validation
    drop_remainder=False,
    field_size=args.field_size,
    norm_val=density_normalization
)

logging.info('val dataset has shape: %s', str(val_dataset))

# ============================
# Build and compile model
# ============================

strategy = tf.distribute.MirroredStrategy()
logging.info(f"Number of devices: {strategy.num_replicas_in_sync}")

logging.info("Building and compiling model...")
logging.info("Preparing model...")

keras.backend.clear_session()

with strategy.scope():
   
    mask_channels = 2 if use_mask else 1

    base_model_obj = UNet(input_field=args.input_field, norm_val=density_normalization)
    base_model_obj.set_logger(logging)
    
    base_model = base_model_obj.prepare_model(
        input_size=(args.field_size, args.field_size, args.field_size, mask_channels)
        )

    base_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate, amsgrad=True, global_clipnorm=1e-2),
        loss=["mean_squared_error"],
        metrics=["mean_squared_error"],
        )
    
logging.info("Model summary:\n%s", str(base_model.summary()))
logging.info("Model built and compiled successfully")

## end of with strategy.scope()
# ============================
# Callbacks
# ============================
from CosmoFiller.checkpoints import EpochCheckpoint
from keras.callbacks import CSVLogger

# personalized callback to save every N epochs
checkpoint_cb = EpochCheckpoint(
    filepath=os.path.join(args.output_dir, 'store_models', 'model_{epoch:02d}.keras'),
    period=args.save_freq  # salva ogni N epoche
)
tensorboard_cb = keras.callbacks.TensorBoard(
    log_dir=os.path.join(args.output_dir, 'logs'),
    histogram_freq=1
    )

# Early stopping to prevent overfitting 
earlystop_cb = keras.callbacks.EarlyStopping(
    monitor='val_loss',      # misura la loss sulla validation set
    patience=20,             # numero di epoche senza miglioramento prima di fermare
    verbose=2,
    restore_best_weights=True  # alla fine ripristina i pesi migliori
)

csv_logger = CSVLogger(os.path.join(args.output_dir, "history.csv"), append=False)

# ============================
# debug prints
# ============================

if args.debug:
    logging.info("Debug mode enabled - printing a sample batch from val_dataset")
    for i, batch in enumerate(val_dataset.take(5)):
        x, y_true = batch
        logging.debug("Batch %d - Input: %f %f", i, tf.reduce_min(x).numpy(), tf.reduce_max(x).numpy())
        logging.debug("Batch %d - Target: %f %f", i, tf.reduce_min(y_true).numpy(), tf.reduce_max(y_true).numpy())
        # Try a prediction
        y_pred = base_model(x, training=False)
        logging.debug("Batch %d - Prediction: %f %f", i, tf.reduce_min(y_pred).numpy(), tf.reduce_max(y_pred).numpy())
        # Manual MSE
        logging.debug("Batch %d - MSE: %f", i, tf.reduce_mean(tf.square(y_true - y_pred)).numpy())

# ============================
# Train
# ============================

steps_per_epoch = int(np.ceil(len(x_train) / args.batch_size)) if args.repeat_dataset else None

logging.info("Starting fit...")

history = base_model.fit(
    train_dataset,
    batch_size=args.batch_size,
    validation_data=val_dataset,
    epochs=args.epochs,
    callbacks=[checkpoint_cb, tensorboard_cb, earlystop_cb, csv_logger],
    steps_per_epoch=steps_per_epoch,  # necessario se il dataset è ripetuto indefinitamente
    verbose=2
)

logging.info("Fit completed")

# ============================
# Save losses
# ============================
logging.info("Saving training and validation losses...")
loss_dir = os.path.join(args.output_dir, 'losses')
os.makedirs(loss_dir, exist_ok=True)
np.save(os.path.join(loss_dir, 'loss.npy'), history.history['loss'])
np.save(os.path.join(loss_dir, 'val_loss.npy'), history.history['val_loss'])
logging.info("Losses saved successfully")
logging.info("Training finished successfully")

# ============================
# Evaluation on validation set and save predicted fields using val_dataset
# ============================
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)
logging.info("Starting evaluation on validation set using val_dataset_eval...")

# Ri-crea val_dataset con batch_size=1 solo per la valutazione
logging.info("Creating evaluation dataset with batch_size=1...")
val_dataset_eval = create_dataset(
    x_valid, y_valid,
    mask_files=mask_valid if mask_valid is not None else None,
    single_mask=single_mask if use_single_mask else None,
    batch_size=1,       # batch singolo per salvare un file per campo
    shuffle=False,      # no shuffle
    repeat=False,       # no repeat
    use_mask=use_mask
)

# predictions is a batch of outputs with shape (batch_size, Nx, Ny, Nz, 1)
logging.info("Predicting on validation dataset...")
predictions = base_model.predict(val_dataset_eval)

for i, pred in enumerate(predictions):
    logging.info(f"Saving predicted field for sample {i+1}/{len(x_valid)}")

    # denormalize for comparison with true and observed fields
    pred_field = pred[..., 0] * density_normalization
    np.save(os.path.join(output_dir, 'output_data', f'pred_field_{i:03d}.npy'), pred_field)


logging.info(f"Saved predicted fields in {output_dir}/output_data/")
logging.info("Evaluation completed successfully")

endtime = datetime.now()  # fine timer
logging.info(f"Total execution time: {endtime - start_time}")
