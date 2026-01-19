#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import json
import numpy as np
import tensorflow as tf
from pathlib import Path

# --- SETUP GPU ---
# Permette di usare la GPU senza allocare tutta la VRAM subito
physical_gpus = tf.config.list_physical_devices('GPU')
if physical_gpus:
    try:
        if len(physical_gpus) >= 2:
            tf.config.set_visible_devices(physical_gpus[2:4], 'GPU')
        else:
            tf.config.set_visible_devices(physical_gpus, 'GPU')
        # Enable memory growth for the (visible) GPUs
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, False)
    except Exception as e:
        # If TF has already initialized GPUs this may fail; warn and continue
        print(f"Warning: could not configure GPUs: {e}")

# --- IMPORTS ---
# Aggiungiamo la cartella corrente al path per trovare i moduli
# Inserisci la root del progetto in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from CosmoFiller.datahandler2d import create_split_datasets
    from CosmoFiller.inpainting2d import MaskedInpaintingUNet2D
except ImportError as e:
    print(f"ERRORE IMPORT: {e}")
    print("Assicurati di avere 'datahandler2d.py' e 'inpainting2d.py' nella cartella CosmoFiller.")
    sys.exit(1)

# -------------------------------
# MAIN SCRIPT 2D
# -------------------------------

h5file = '/home/emaragliano/Work/Projects/myfarm-disk/EuclidDithering/training_set_quijote_dithering_100mocks_5rot.h5'
x_key = 'inputs'
y_key = 'targets'
output_dir = '/home/emaragliano/Work/Projects/myfarm-disk/AE_storage_Paper2/training_runs/dithering_inpainting'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 2D Inpainting U-Net from HDF5")
    
    # Argomenti Dataset
    parser.add_argument('--h5_file', type=str, required=False, default=h5file, help='Path del file .h5')
    parser.add_argument('--x_key', type=str, default=x_key, help='Nome dataset input nel file h5')
    parser.add_argument('--y_key', type=str, default=y_key, help='Nome dataset target nel file h5')
    parser.add_argument('--output_dir', type=str, default=output_dir, help='Cartella output')
    
    # Iperparametri
    parser.add_argument('--field_size', type=int, default=1024, help='Lato immagine (es. 128)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('--norm_val', type=float, default=40.0, help='Fattore di normalizzazione')
    
    args = parser.parse_args()

    # --- Creazione Cartelle ---
    os.makedirs(args.output_dir, exist_ok=True)
    models_dir = os.path.join(args.output_dir, 'models')
    logs_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # --- Logger ---
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(logs_dir, 'training.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Avvio training 2D...")

    # --- 1. DATASET (Streaming H5) ---
    logger.info(f"Caricamento dataset da: {args.h5_file}")
    
    # Questa funzione fa tutto: legge H5, fa split indici, crea generatori TF
    train_ds, val_ds, test_ds = create_split_datasets(
        h5_path=args.h5_file,
        batch_size=args.batch_size,
        keys=(args.x_key, args.y_key),
        val_split=0.15,   # 15% validazione
        test_split=0.15,  # 15% test
        field_size=args.field_size,
        dim=2,            # FORZIAMO 2D
        seed=42
    )

    # Controllo veloce dimensioni
    try:
        sample_x, sample_y = next(iter(train_ds))
        logger.info(f"Check Batch -> X: {sample_x.shape}, Y: {sample_y.shape}")
        # shape attesa: (Batch, 128, 128, 1)
    except Exception as e:
        logger.error(f"Impossibile leggere dal dataset. Controlla le chiavi H5. Errore: {e}")
        sys.exit(1)

    # --- 2. MODELLO (Multi-GPU ready) ---
    strategy = tf.distribute.MirroredStrategy()
    logger.info(f"GPU in uso: {strategy.num_replicas_in_sync}")

    with strategy.scope():
        # Inizializziamo la classe wrapper 2D
        wrapper = MaskedInpaintingUNet2D(
            input_size=(args.field_size, args.field_size, 1), # 1 canale
            base_filters=16,
            dropout_layer=False,
            norm_val=args.norm_val
        )
        
        model = wrapper.unet
        
        # Compilazione
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
            loss='mse', 
            metrics=['mae']
        )

    # --- 3. TRAINING ---
    logger.info("Inizio training...")
    
    callbacks = [
        # Salva solo il modello migliore sul validation set
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(models_dir, 'best_model.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        # Ferma il training se non migliora 
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=80,
            restore_best_weights=True,
            verbose=1
        ),
        # Log su CSV per fare i grafici dopo
        tf.keras.callbacks.CSVLogger(os.path.join(logs_dir, 'history.csv'))
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )

    logger.info("Training completato.")

    # --- 4. TEST FINALE ---
    logger.info("Valutazione sul Test Set...")
    test_loss = model.evaluate(test_ds)
    logger.info(f"Test Loss (MAE): {test_loss[0]}")

    # Salviamo qualche predizione di esempio
    logger.info("Salvataggio predizioni di esempio...")
    test_sample_x, test_sample_y = next(iter(test_ds)) # Prende un batch dal test
    preds = model.predict(test_sample_x)
    
    np.save(os.path.join(args.output_dir, 'test_input.npy'), test_sample_x.numpy())
    np.save(os.path.join(args.output_dir, 'test_truth.npy'), test_sample_y.numpy())
    np.save(os.path.join(args.output_dir, 'test_pred.npy'), preds)
    
    logger.info(f"Finito. Controlla {args.output_dir}")