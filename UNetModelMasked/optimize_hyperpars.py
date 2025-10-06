#!/usr/bin/env python3
import os
import glob
import json
import logging
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from datetime import datetime
import optuna
from InpaintingModel import InpaintingModel

# =========================================
# Parametri generali
# =========================================
OBS_DIR = "path_to_obs"
TRUE_DIR = "path_to_true"
MASK_DIR = "path_to_masks"  # o None
INPUT_FIELD = "rho"  # 'rho' o 'delta'
FIELD_SIZE = 128
EPOCHS_PER_TRIAL = 5  # epoche brevi per Optuna
REPEAT_DATASET = False
DROP_REMAINDER = False
TEST_SIZE = 0.2

OUTPUT_DIR = "optuna_results/" + datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================
# Setup logging
# =========================================
logging.basicConfig(
    level=logging.INFO,
    
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout), 
              logging.FileHandler(os.path.join(OUTPUT_DIR, "optuna.log"))],
)

logging.info("Starting hyperparameter optimization with Optuna")
logging.info(f"OBS_DIR: {OBS_DIR}", f", TRUE_DIR: {TRUE_DIR}, MASK_DIR: {MASK_DIR}")
logging.info(f"INPUT_FIELD: {INPUT_FIELD}, FIELD_SIZE: {FIELD_SIZE}, EPOCHS_PER_TRIAL: {EPOCHS_PER_TRIAL}")
logging.info(f"REPEAT_DATASET: {REPEAT_DATASET}, DROP_REMAINDER: {DROP_REMAINDER}, TEST_SIZE: {TEST_SIZE}")

# =========================================
# Load file lists
# =========================================
x_files = sorted(glob.glob(os.path.join(OBS_DIR, '*.npy')))
y_files = sorted(glob.glob(os.path.join(TRUE_DIR, '*.npy')))
mask_files = sorted(glob.glob(os.path.join(MASK_DIR, '*.npy'))) if MASK_DIR else None

assert len(x_files) == len(y_files), "Mismatch between obs and true files!"

# =========================================
# Density normalization
# =========================================
density_normalization = 40.0 if INPUT_FIELD == "rho" else 20.0

# =========================================
# Split dataset
# =========================================
x_train, x_valid, y_train, y_valid = train_test_split(x_files, y_files, test_size=TEST_SIZE, shuffle=True)
mask_train = mask_valid = None
use_mask = mask_files is not None and len(mask_files) > 0
single_mask = None

if use_mask:
    if len(mask_files) == 1:
        single_mask = np.load(mask_files[0]).astype(np.float32)
    else:
        mask_train, mask_valid = train_test_split(mask_files, test_size=TEST_SIZE, shuffle=True)

# =========================================
# Dataset parsing
# =========================================
def parse_fn(obs_path, true_path, mask_path=None, single_mask=None, use_mask=True):
    obs = tf.numpy_function(lambda f: np.load(f).astype(np.float32)/density_normalization, [obs_path], tf.float32)
    true = tf.numpy_function(lambda f: np.load(f).astype(np.float32)/density_normalization, [true_path], tf.float32)
    obs = tf.expand_dims(obs, axis=-1)
    true = tf.expand_dims(true, axis=-1)

    if use_mask:
        if single_mask is not None:
            mask = tf.convert_to_tensor(single_mask, dtype=tf.float32)
        else:
            mask = tf.numpy_function(lambda f: np.load(f).astype(np.float32), [mask_path], tf.float32)
        mask = tf.expand_dims(mask, axis=-1)
        x = tf.concat([obs, mask], axis=-1)
        x.set_shape((FIELD_SIZE, FIELD_SIZE, FIELD_SIZE, 2))
        true.set_shape((FIELD_SIZE, FIELD_SIZE, FIELD_SIZE, 1))
        mask.set_shape((FIELD_SIZE, FIELD_SIZE, FIELD_SIZE, 1))
        return x, true
    else:
        obs.set_shape((FIELD_SIZE, FIELD_SIZE, FIELD_SIZE, 1))
        true.set_shape((FIELD_SIZE, FIELD_SIZE, FIELD_SIZE, 1))
        return obs, true

def create_dataset(x_files, y_files, mask_files=None, single_mask=None, batch_size=16,
                   shuffle=True, use_mask=True, repeat=False, drop_remainder=False):
    if use_mask and single_mask is not None:
        dataset = tf.data.Dataset.from_tensor_slices((x_files, y_files))
        dataset = dataset.map(lambda x,y: parse_fn(x,y,single_mask=single_mask,use_mask=True),
                              num_parallel_calls=tf.data.AUTOTUNE)
    elif use_mask and mask_files is not None:
        dataset = tf.data.Dataset.from_tensor_slices((x_files, y_files, mask_files))
        dataset = dataset.map(lambda x,y,m: parse_fn(x,y,mask_path=m,use_mask=True),
                              num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = tf.data.Dataset.from_tensor_slices((x_files, y_files))
        dataset = dataset.map(lambda x,y: parse_fn(x,y,use_mask=False),
                              num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(x_files), reshuffle_each_iteration=True)
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder).prefetch(tf.data.AUTOTUNE)
    return dataset

# =========================================
# Funzione di training per un trial Optuna
# =========================================
def train_model(batch_size, learning_rate):
    train_dataset = create_dataset(x_train, y_train,
                                   mask_files=mask_train,
                                   single_mask=single_mask,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   use_mask=use_mask,
                                   repeat=False,
                                   drop_remainder=DROP_REMAINDER)
    
    val_dataset = create_dataset(x_valid, y_valid,
                                 mask_files=mask_valid,
                                 single_mask=single_mask,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 use_mask=use_mask,
                                 repeat=False,
                                 drop_remainder=DROP_REMAINDER)
    
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        mask_channels = 2 if use_mask else 1
        base_model_obj = InpaintingModel(input_field=INPUT_FIELD, norm_val=density_normalization)
        model = base_model_obj.prepare_model(
            input_size=(FIELD_SIZE, FIELD_SIZE, FIELD_SIZE, mask_channels)
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=True, global_clipnorm=1e-2),
            loss="mean_squared_error",
            metrics=["mean_squared_error"]
        )
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS_PER_TRIAL,
        verbose=0
    )
    
    # Restituisci la val_loss finale
    return history.history['val_loss'][-1]

# =========================================
# Ottimizzazione con Optuna
# =========================================
def objective(trial):
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    val_loss = train_model(batch_size, learning_rate)
    print(f"Trial {trial.number}: batch_size={batch_size}, lr={learning_rate}, val_loss={val_loss:.6f}")
    return val_loss

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

print("Best trial:")
trial = study.best_trial
print(f"Batch size: {trial.params['batch_size']}")
print(f"Learning rate: {trial.params['learning_rate']}")


# Salva i risultati
if not os.path.exists('optuna_study'):
    os.makedirs('optuna_study')
with open('optuna_study/best_trial.json', 'w') as f:
    json.dump({'learning_rate': trial.params['learning_rate'],
               'batch_size': trial.params['batch_size'],
               'val_loss': trial.value}, f)
with open('optuna_study/study_summary.txt', 'w') as f:
    f.write(f"Number of trials: {len(study.trials)}\n")
    f.write(f"Best trial:\n")
    f.write(f"  Learning rate: {trial.params['learning_rate']}\n")
    f.write(f"  Batch size: {trial.params['batch_size']}\n")
    f.write(f"  Validation loss: {trial.value}\n")