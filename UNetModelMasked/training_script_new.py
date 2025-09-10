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
from sklearn.model_selection import train_test_split
from InpaintingModel import InpaintingModel
from datetime import datetime

start_time = datetime.now()  # inizio timer

# ============================
# Command-line arguments
# ============================
parser = argparse.ArgumentParser(description='Train 3D U-Net for inpainting.')
parser.add_argument('--param_file', type=str, help='Path to JSON parameter file')
parser.add_argument('--obs_dir', type=str, help='Directory with observed .npy fields')
parser.add_argument('--true_dir', type=str, help='Directory with true .npy fields')
parser.add_argument('--mask_dir', type=str, help='Directory with mask .npy files')
parser.add_argument('--input_field', type=str, default='rho', help='Type of input field: delta or rho')
parser.add_argument('--output_dir', type=str, default='output_products', help='Directory to store output fields')
parser.add_argument('--use_mask', type=bool, default=False, help='True = use mask, False = do not use mask')
parser.add_argument('--field_size', type=int, default=128, help='Size of input fields (NxNxN)')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--save_freq', type=int, default=10, help='Frequency (in epochs) to save the model')
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--log_file', type=str, default='logs/training.log', help='File to store logs')
parser.add_argument('--debug', action='store_true', help='Enable debug mode with verbose logging')


args = parser.parse_args()

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
if args.input_field == 'delta':
    density_normalization = 20.0
elif args.input_field == 'rho':
    density_normalization = 40.0


# ============================
# Setup logging
# ============================
# Clear existing handlers for the root logger
for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)

# Main logger
logging.basicConfig(
    level=logging.INFO if not args.debug else logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(args.log_file),
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)


# ============================
# Collect files
# ============================
x_files = sorted(glob.glob(os.path.join(args.obs_dir,
             'zeldovich_rec_density_*_from_redshift_space_128_CIC_sm_10_zeldareco.npy')))
y_files = sorted(glob.glob(os.path.join(args.true_dir, 
            'halo_number_density_real_space_*_128_CIC_group_tab_002.npy')))
mask_files = sorted(glob.glob(os.path.join(args.mask_dir, '*.npy')))

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

# ============================
# Data parsing functions
# ============================
def parse_fn(obs_path, true_path, mask_path=None, single_mask=None, use_mask=True):
    obs = tf.numpy_function(lambda f: np.load(f).astype(np.float32)/density_normalization, [obs_path], tf.float32)
    true = tf.numpy_function(lambda f: np.load(f).astype(np.float32)/density_normalization, [true_path], tf.float32)
    obs = tf.expand_dims(obs, axis=-1)
    true = tf.expand_dims(true, axis=-1)

    if use_mask:
        if single_mask is not None:
            # Use the provided single mask for all samples
            mask = tf.convert_to_tensor(single_mask, dtype=tf.float32)
            logging.debug("Using single mask with shape: %s", str(mask.shape))
            # Expand dimensions for tf compatibility
            mask = tf.expand_dims(mask, axis=-1)
            logging.debug("Expanded single mask shape: %s", str(mask.shape))
        else:
            # Load different masks for each sample
            mask = tf.numpy_function(lambda f: np.load(f), [mask_path], tf.float32)
            logging.debug("Loaded mask from %s with shape: %s", mask_path.numpy(), str(mask.shape))
            # Expand dimensions for tf compatibility
            mask = tf.expand_dims(mask, axis=-1)
            logging.debug("Expanded mask shape: %s", str(mask.shape))
        x = tf.concat([obs, mask], axis=-1)
        x.set_shape((args.field_size, args.field_size, args.field_size, 2))
        logging.debug("Input and mask concatenated. Shape is: %s", str(x.shape))
        logging.debug("Channel 1 is obs, channel 2 is mask")
        true.set_shape((args.field_size, args.field_size, args.field_size, 1))
        mask.set_shape((args.field_size, args.field_size, args.field_size, 1))
        return (x, true, mask)
    else:
        x = obs
        logging.debug("Single channel input (obs only)")
        x.set_shape((args.field_size, args.field_size, args.field_size, 1))
        true.set_shape((args.field_size, args.field_size, args.field_size, 1))
        return (x, true)

def create_dataset(x_files, y_files, mask_files=None, single_mask=None, batch_size=4, shuffle=True, use_mask=True):

    logging.debug("create_dataset called with: %d x_files, %d y_files, mask_files=%s, single_mask=%s, batch_size=%d, shuffle=%s, use_mask=%s", 
        len(x_files), len(y_files), str(mask_files is not None), str(single_mask is not None), batch_size, str(shuffle), str(use_mask))
    
    if use_mask and single_mask is not None:
        # create dataset with single mask for all samples
        dataset = tf.data.Dataset.from_tensor_slices((x_files, y_files))
        logging.debug("Dataset created from x_files and y_files only. Mapping with single_mask.")
        # add the same mask to all samples in the dataset
        dataset = dataset.map(lambda x,y: parse_fn(x,y,single_mask=single_mask,use_mask=True),\
                               num_parallel_calls=tf.data.AUTOTUNE)

    elif use_mask and mask_files is not None:
        # create dataset with individual masks for each sample
        dataset = tf.data.Dataset.from_tensor_slices((x_files, y_files, mask_files))
        logging.debug("Dataset created from x_files, y_files, and mask_files. Mapping with mask_path.")
        dataset = dataset.map(lambda x,y,m: parse_fn(x,y,mask_path=m,use_mask=True),\
                               num_parallel_calls=tf.data.AUTOTUNE)
    
    else:
        # create dataset without masks
        dataset = tf.data.Dataset.from_tensor_slices((x_files, y_files))
        logging.debug("Dataset created from x_files and y_files only. Mapping without mask.")
        dataset = dataset.map(lambda x,y: parse_fn(x,y,use_mask=False),\
                               num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        logging.debug("Shuffling dataset with buffer size %d", min(len(x_files), 16))
        dataset = dataset.shuffle(buffer_size=len(x_files), reshuffle_each_iteration=shuffle)
    else:
        logging.debug("No shuffle applied to dataset.")

    logging.debug("Batching dataset with batch_size=%d", batch_size)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    logging.debug("Dataset ready. Returning tf.data.Dataset object.")

    return dataset


train_dataset = create_dataset(
    x_train, y_train,
    mask_files=mask_train if mask_train is not None else None,
    single_mask=single_mask if use_single_mask else None,
    batch_size=args.batch_size,
    use_mask=use_mask,
    shuffle=True,
    )

logging.info('train dataset has shape: %s', str(train_dataset))

val_dataset = create_dataset(
    x_valid, y_valid,
    mask_files=mask_valid if mask_valid is not None else None,
    single_mask=single_mask if use_single_mask else None,
    batch_size=args.batch_size,
    shuffle=False,                  # No shuffle for validation
    use_mask=use_mask
)

logging.info('val dataset has shape: %s', str(val_dataset))

# ============================
# Masked loss
# ============================
def masked_mse_loss(y_true, y_pred, mask=None):
    if mask is None:
        # MSE classico su tutto il volume
        return tf.reduce_mean(tf.square(y_true - y_pred))
    else:
        # MSE mascherato solo sulle regioni missing
        missing_mask = 1.0 - mask
        squared_error = tf.square(y_true - y_pred)
        masked_error = squared_error * missing_mask
        # Se la maschera è tutta 1, la loss sarà zero (nessuna regione da ricostruire)
        return tf.reduce_sum(masked_error) / (tf.reduce_sum(missing_mask) + 1e-8)

# ============================
# Custom training model (commented out for standard MSE)
# ============================
# class MaskedInpaintingModel(keras.Model):
#     def train_step(self, data):
#         """Custom training step with masked loss or standard MSE."""
#         if use_mask:
#             x, y_true, mask = data
#             with tf.GradientTape() as tape:
#                 y_pred = self(x, training=True)
#                 loss = masked_mse_loss(y_true, y_pred, mask)
#             grads = tape.gradient(loss, self.trainable_variables)
#             self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
#             return {"loss": loss}
#         else:
#             x, y_true = data
#             with tf.GradientTape() as tape:
#                 y_pred = self(x, training=True)
#                 loss = masked_mse_loss(y_true, y_pred)
#             grads = tape.gradient(loss, self.trainable_variables)
#             self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
#             return {"loss": loss}
#
#     def test_step(self, data):
#         """Custom validation step with masked loss or standard MSE."""
#         if use_mask:
#             x, y_true, mask = data
#             y_pred = self(x, training=False)
#             val_loss = masked_mse_loss(y_true, y_pred, mask)
#             return {"loss": val_loss}
#         else:
#             x, y_true = data
#             y_pred = self(x, training=False)
#             val_loss = masked_mse_loss(y_true, y_pred)
#             return {"loss": val_loss}

# ============================
# Build and compile model
# ============================

#strategy = tf.distribute.MirroredStrategy()
#logging.info("Number of devices: {}".format(strategy.num_replicas_in_sync))

#with strategy.scope():
keras.backend.clear_session()
logging.info("Building and compiling model...")

mask_channels = 2 if use_mask else 1

base_model_obj = InpaintingModel(input_field=args.input_field, norm_val=density_normalization)
base_model_obj.set_logger(logging)
logging.info("Preparing model...")
base_model = base_model_obj.prepare_model(
    input_size=(args.field_size, args.field_size, args.field_size, mask_channels)
    )
logging.info("Model summary:\n%s", str(base_model.summary()))
# masked_model = MaskedInpaintingModel(inputs=base_model.input, outputs=base_model.output)
base_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate, amsgrad=True, global_clipnorm=1e-2),
    loss=["mean_squared_error"],
    metrics=["mean_squared_error"],
    )
logging.info("Model built and compiled successfully")

# ============================
# Callbacks
# ============================

checkpoint_cb = keras.callbacks.ModelCheckpoint(
    filepath='./store_models/model_{epoch:02d}.keras',
    monitor='loss',
    save_freq='epoch',
    save_weights_only=False,
    save_best_only=False
)
tensorboard_cb = keras.callbacks.TensorBoard(
    log_dir='./logs',
    histogram_freq=1
    )


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

logging.info("Starting fit...")
history = base_model.fit(
    train_dataset,
    batch_size=args.batch_size,
    validation_data=val_dataset,
    epochs=args.epochs,
    callbacks=[checkpoint_cb, tensorboard_cb]
)
logging.info("Fit completed")

# ============================
# Save losses
# ============================
logging.info("Saving training and validation losses...")
os.makedirs('./losses', exist_ok=True)
np.save('./losses/loss.npy', history.history['loss'])
np.save('./losses/val_loss.npy', history.history['val_loss'])
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
    shuffle=False,
    use_mask=use_mask
)

# predictions is a batch of outputs with shape (batch_size, Nx, Ny, Nz, 1)
logging.info("Predicting on validation dataset...")
predictions = base_model.predict(val_dataset_eval)

for i, pred in enumerate(predictions):
    logging.info(f"Saving predicted field for sample {i+1}/{len(x_valid)}")
    # denormalize for comparison with true and observed fields
    pred_field = pred[..., 0] * density_normalization
    np.save(os.path.join(output_dir, f'pred_field_{i:03d}.npy'), pred_field)
    #np.save(os.path.join(output_dir, f'true_field_{i:03d}.npy'), np.load(y_valid[i]))
    #np.save(os.path.join(output_dir, f'obs_field_{i:03d}.npy'), np.load(x_valid[i]))

logging.info(f"Saved predicted fields in {output_dir}")
logging.info("Evaluation completed successfully")

endtime = datetime.now()  # fine timer
logging.info(f"Total execution time: {endtime - start_time}")
