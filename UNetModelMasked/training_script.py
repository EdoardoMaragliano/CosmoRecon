#!/usr/bin/env python3
import os
import glob
import argparse
import logging
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from InpaintingModel import InpaintingModel
from sklearn.model_selection import train_test_split

# ============================
# Command-line arguments
# ============================
parser = argparse.ArgumentParser(description='Train 3D U-Net for inpainting.')
parser.add_argument('--param_file', type=str, help='Path to JSON parameter file')
parser.add_argument('--obs_dir', type=str, help='Directory with observed .npy fields')
parser.add_argument('--true_dir', type=str, help='Directory with true .npy fields')
parser.add_argument('--mask_dir', type=str, help='Directory with mask .npy files')
parser.add_argument('--use_mask', type=bool, default=False, help='True = use mask, False = do not use mask')
parser.add_argument('--field_size', type=int, default=128, help='Size of input fields (NxNxN)')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--log_file', type=str, default='logs/training.log', help='File to store logs')
parser.add_argument('--debug', action='store_true', help='Enable debug mode with verbose logging')

args = parser.parse_args()

# ============================
# Load parameters from JSON if provided
# ============================
if args.param_file:
    
    with open(args.param_file, 'r') as f:
        params = json.load(f)
    for key, value in params.items():
        if hasattr(args, key):
            setattr(args, key, value)

use_mask = bool(args.use_mask)

# Normalization factor for density fields
density_normalization = 40.0

# ============================
# Setup logging
# ============================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(args.log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

# Redirect TensorFlow logging to our logger
tf.get_logger().setLevel('INFO')

logging.info("Starting training script")
if args.param_file:
    logging.info(f"Loading parameters from {args.param_file}")

logging.info(f"Observed data dir: {args.obs_dir}")
logging.info(f"True data dir: {args.true_dir}")
logging.info(f"Mask data dir: {args.mask_dir}")
logging.info(f"Field size: {args.field_size}")
logging.info(f"Batch size: {args.batch_size}")
logging.info(f"Epochs: {args.epochs}")
logging.info(f"Learning rate: {args.learning_rate}")
logging.info(f"Use mask: {use_mask}\n\n")

# ============================
# Automatically collect files
# ============================
x_files = sorted(glob.glob(os.path.join(args.obs_dir, '*.npy')))
y_files = sorted(glob.glob(os.path.join(args.true_dir, '*.npy')))
mask_files = sorted(glob.glob(os.path.join(args.mask_dir, '*.npy')))

assert len(x_files) == len(y_files), "Mismatch in number of observed and true files!"

# Single mask check
if use_mask and mask_files:
    if len(mask_files) == 1:
        single_mask = np.load(mask_files[0]).astype(np.float32)
        single_mask = np.expand_dims(single_mask, axis=-1) if len(single_mask.shape)==3 else single_mask
        use_single_mask = True
        logging.info("Using a single mask for all samples")
    else:
        single_mask = None
        use_single_mask = False
        assert len(mask_files) == len(x_files), "Mismatch in number of mask files!"
        logging.info(f"Using individual masks for each sample ({len(mask_files)} files)")
else:
    single_mask = None
    use_single_mask = False
    use_mask = False
    logging.info("No mask will be used")

logging.info(f"Found {len(x_files)} files for training")

# ============================
# Split dataset
# ============================

if use_mask and use_single_mask:
    x_train, x_valid, y_train, y_valid = train_test_split(x_files, y_files, test_size=0.2, shuffle=False)
    mask_train = mask_valid = None
elif use_mask and not use_single_mask:
    x_train, x_valid, y_train, y_valid, mask_train, mask_valid = train_test_split(
        x_files, y_files, mask_files, test_size=0.2, shuffle=False)
else:
    x_train, x_valid, y_train, y_valid = train_test_split(x_files, y_files, test_size=0.2, shuffle=False)
    mask_train = mask_valid = None

logging.info("Training and validation datasets created successfully.")
logging.info(f"Training samples: {len(x_train)}, Validation samples: {len(x_valid)}")


# ============================
# Helper function to load data
# ============================
def parse_fn(obs_path, true_path, mask_path=None, single_mask=None, use_mask=True):
    """
    Parse function for loading and preprocessing data.
    """

    # Normalization
    obs = tf.numpy_function(lambda f: np.load(f)/density_normalization, [obs_path], tf.float32)
    true = tf.numpy_function(lambda f: np.load(f)/density_normalization, [true_path], tf.float32)
    # Expand dimensions
    obs = tf.expand_dims(obs, axis=-1)
    true = tf.expand_dims(true, axis=-1)

    if use_mask:    # Check if mask should be used
        if single_mask is not None: # Check if using a single mask 
            mask = tf.convert_to_tensor(single_mask, dtype=tf.float32)
            mask = tf.expand_dims(mask, axis=-1) if len(mask.shape)==3 else mask
        else:   
            mask = tf.numpy_function(lambda f: np.load(f), [mask_path], tf.float32)
            mask = tf.expand_dims(mask, axis=-1)
        x = tf.concat([obs, mask], axis=-1)
    else:
        x = obs

    # Set final shape
    x.set_shape((args.field_size, args.field_size, args.field_size, 2 if use_mask else 1))
    return x, true, mask if use_mask else None


def create_dataset(x_files, y_files, mask_files=None, single_mask=None, batch_size=4, shuffle=True, use_mask=True):
    if use_mask and single_mask is not None:
        dataset = tf.data.Dataset.from_tensor_slices((x_files, y_files))
        dataset = dataset.map(lambda x,y: parse_fn(x,y,single_mask=single_mask,use_mask=True), num_parallel_calls=tf.data.AUTOTUNE)
    elif use_mask and mask_files is not None:
        dataset = tf.data.Dataset.from_tensor_slices((x_files, y_files, mask_files))
        dataset = dataset.map(lambda x,y,m: parse_fn(x,y,mask_path=m,use_mask=True), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = tf.data.Dataset.from_tensor_slices((x_files, y_files))
        dataset = dataset.map(lambda x,y: parse_fn(x,y,use_mask=False), num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(x_files), 16))

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# ============================
# Create tf.data.Dataset
# ============================
train_dataset = create_dataset(
    x_train, y_train,
    mask_files=mask_train if mask_train is not None else None,
    single_mask=single_mask if use_single_mask else None,
    batch_size=args.batch_size,
    use_mask=use_mask
)

val_dataset = create_dataset(
    x_valid, y_valid,
    mask_files=mask_valid if mask_valid is not None else None,
    single_mask=single_mask if use_single_mask else None,
    batch_size=args.batch_size,
    shuffle=False,
    use_mask=use_mask
)

for x_val, y_val, mask_val in val_dataset.take(1):
    logging.info(f"Validation shapes - x: {x_val.shape}, y: {y_val.shape}, mask: {mask_val.shape if mask_val is not None else None}")


# ============================
# Masked MSE loss
# ============================
def masked_mse_loss(y_true, y_pred, mask=None):
    if mask is None:
        return tf.reduce_mean(tf.square(y_true - y_pred))
    missing_mask = 1.0 - mask
    squared_error = tf.square(y_true - y_pred)
    masked_error = squared_error * missing_mask
    return tf.reduce_sum(masked_error) / (tf.reduce_sum(missing_mask) + 1e-8)


# ============================
# Custom training model
# ============================

"""
Inpainting may require computing the loss only on missing voxels (where mask = 0).
Keras does not natively allow passing extra tensors (like a mask) to the loss function when \
    using Model.fit in standard mode.
To fix this, we subclass keras.Model and override train_step (and optionally test_step):
base_model → defines architecture.
masked_model → defines training behavior (masked loss).
"""
class MaskedInpaintingModel(keras.Model):
    def train_step(self, data):
        """
        Custom training step for the masked inpainting model.
        """
        x, y_true, mask = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)                     # self is the masked model instance
            loss = masked_mse_loss(y_true, y_pred, mask)        # call the masked loss
        
        # gradient and optimization
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": loss}

    def test_step(self, data):
        x, y_true, mask = data
        y_pred = self(x, training=False)
        loss = masked_mse_loss(y_true, y_pred, mask)
        return {"loss": loss}

# ============================
# Build and compile model
# ============================
mask_channels = 2 if use_mask else 1

#define the model architecture
base_model = InpaintingModel().prepare_model(input_size=(args.field_size, args.field_size, args.field_size, mask_channels))
#train with masked loss
masked_model = MaskedInpaintingModel(inputs=base_model.input, outputs=base_model.output)
#compile the masked model
masked_model.compile(optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate))

# ============================
# Callbacks
# ============================
checkpoint_cb = keras.callbacks.ModelCheckpoint('./store_models/model_{epoch:02d}.keras', save_freq='epoch')
tensorboard_cb = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1)

# ============================
# Train
# ============================
history = masked_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=args.epochs,
    callbacks=[checkpoint_cb, tensorboard_cb]
)

# ============================
# Save losses
# ============================
np.save('./losses/loss.npy', history.history['loss'])
np.save('./losses/val_loss.npy', history.history['val_loss'])
logging.info("Training finished successfully")


