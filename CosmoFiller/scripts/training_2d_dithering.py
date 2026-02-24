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
            tf.config.set_visible_devices(physical_gpus[0:2], 'GPU')
        else:
            tf.config.set_visible_devices(physical_gpus[1], 'GPU')
        # Enable memory growth for the (visible) GPUs
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
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
COORDSYS = 'equatorial'  # 'ecliptic' o 'equatorial'
h5file = f'/home/emaragliano/Work/Projects/Dottorato/EuclidInpaintingNew/density_maps/dataset_quijote_z1_512_{COORDSYS}_downsampled_5_rot.h5'
x_key = 'inputs'
y_key = 'targets'

W_MAP = 1.0
W_RES = 10.0

class LossWeightScheduler(tf.keras.callbacks.Callback):
    def __init__(self, w_res_tensor, schedule_fn):
        super().__init__()
        self.w_res_tensor = w_res_tensor
        self.schedule_fn = schedule_fn

    def on_epoch_begin(self, epoch, logs=None):
        # Calcola il nuovo valore in base all'epoca
        new_w = self.schedule_fn(epoch)
        # Aggiorna il valore del tensore variabile
        tf.keras.backend.set_value(self.w_res_tensor, new_w)
        print(f"\n[Weight Scheduler] Epoch {epoch}: W_RES aggiornato a {new_w:.4f}")

# --- Esempio di funzione schedule (es. crescita lineare) ---
def my_schedule(epoch):
    initial_w = 1.0
    final_w = 50.0
    warmup_epochs = 100
    if epoch < warmup_epochs:
        return initial_w + (final_w - initial_w) * (epoch / warmup_epochs)
    return final_w

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 2D Inpainting U-Net from HDF5")
    
    # Argomenti Dataset
    parser.add_argument('--h5_file', type=str, required=False, default=h5file, help='Path del file .h5')
    parser.add_argument('--x_key', type=str, default=x_key, help='Nome dataset input nel file h5')
    parser.add_argument('--y_key', type=str, default=y_key, help='Nome dataset target nel file h5')
    parser.add_argument('--output_dir', type=str, default='training_runs/out', help='Cartella output')
    
    # Iperparametri
    parser.add_argument('--field_size', type=int, default=512, help='Lato immagine (es. 128)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('--norm_val', type=float, default=10.0, help='Fattore di normalizzazione')
    parser.add_argument('--loss', type=str, default='mse', help='Funzione di loss da usare')
    parser.add_argument('--residuals_only', action='store_true', help='Usa apprendimento residuo (output = input + residuo)')
    parser.add_argument('--augment', action='store_true', help='Abilita data augmentation (rotazioni e flip)')
    
    args = parser.parse_args()

    # override output_dir per includere i parametri chiave (per organizzazione)
    if args.residuals_only:
        output_dir = '/home/emaragliano/Work/Projects/myfarm-disk/AE_storage_Paper2/' \
        f'training_runs/dithering_quijote_5rot_{args.epochs}_epochs_{args.field_size}_{COORDSYS}_downsampled_residuals_only_norm{args.norm_val}_WMAP_{W_MAP}_WRES_{W_RES}_scheduled'
        args.output_dir = output_dir
    else:
        output_dir = '/home/emaragliano/Work/Projects/myfarm-disk/AE_storage_Paper2/' \
        f'training_runs/dithering_quijote_5rot_{args.epochs}_epochs_{args.field_size}_{COORDSYS}_downsampled_norm{args.norm_val}_loss_{args.loss}'
        args.output_dir = output_dir

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
    logger.info(f"Parametri: {json.dumps(vars(args), indent=2)}")

    logger.info("W_MAP: {:.1f}, W_RES: {:.1f}".format(W_MAP, W_RES))

    # --- 1. DATASET (Streaming H5) ---
    logger.info(f"Caricamento dataset da: {args.h5_file}")
    
    # Questa funzione fa tutto: legge H5, fa split indici, crea generatori TF
    train_ds, val_ds, test_ds = create_split_datasets(
        h5_path=args.h5_file,
        batch_size=args.batch_size,
        keys=(args.x_key, args.y_key),
        val_split=0.1,          # 10% validazione
        test_split=0.1,         # 10% test
        norm_val=args.norm_val,
        field_size=args.field_size,
        dim=2,            # FORZIAMO 2D
        seed=42
    )

    def augment(x, y):
        # Esempio: rotazione casuale multipla di 90 gradi
        k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
        x = tf.image.rot90(x, k)
        y = tf.image.rot90(y, k)
        # Flip casuale
        #x = tf.image.random_flip_left_right(x)         #peggiora molto il training
        #y = tf.image.random_flip_left_right(y)         #peggiora molto il training
        return x, y
    
    def prepare_residual_target(x, y):
        # x è ndith_ge_4, y è ndith_ge_1
        # Il modello ha due output: 'final_map' e 'residual_layer'
        residual = y - x
        return x, {'final_map': y, 'residual_layer': residual}

    if args.augment:
        logger.info("Abilitata data augmentation (rotazioni e flip)")
        train_ds = train_ds.map(augment)

    if args.residuals_only:
        train_ds = train_ds.map(prepare_residual_target)
        val_ds = val_ds.map(prepare_residual_target)
        test_ds = test_ds.map(prepare_residual_target)

    # Controllo veloce dimensioni
    try:
        sample_x, sample_y = next(iter(train_ds))
        
        # shape attesa: (Batch, IMG_SIZE, IMG_SIZE, 1)
        # Se residuals_only è True, sample_y è un dict, altrimenti è un Tensor
        if isinstance(sample_y, dict):
            y_shape = sample_y['final_map'].shape
        else:
            y_shape = sample_y.shape
        logger.info(f"Check Batch -> X: {sample_x.shape}, Y: {y_shape}")
        
    except Exception as e:
        logger.error(f"Impossibile leggere dal dataset. Controlla le chiavi H5. Errore: {e}")
        sys.exit(1)

    # --- 2. MODELLO (Multi-GPU ready) ---
    strategy = tf.distribute.MirroredStrategy()
    logger.info(f"GPU in uso: {strategy.num_replicas_in_sync}")

    with strategy.scope():

        # Definiamo i pesi come variabili mutabili
        W_RES_VAR = tf.Variable(W_RES, dtype=tf.float32, trainable=False)
        W_MAP_VAR = tf.Variable(W_MAP, dtype=tf.float32, trainable=False)

        # Inizializziamo la classe wrapper 2D
        wrapper = MaskedInpaintingUNet2D(
            input_size=(args.field_size, args.field_size, 1), # 1 canale
            base_filters=16,
            dropout_layer=False,
            norm_val=args.norm_val,
            residuals_only=args.residuals_only
        )
        
        model = wrapper.unet
        
        LOSS_DICT = {
            'mse': tf.keras.losses.MeanSquaredError(),
            'mae': tf.keras.losses.MeanAbsoluteError(),
            'huber': tf.keras.losses.Huber(delta=0.005)
        }
        # Compilazione
        if not args.residuals_only:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                loss=LOSS_DICT[args.loss], 
                metrics=['mae', 'mse']
            )
        else:
            # Se usiamo residuals_only, la loss è solo sul residuo
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                loss={
                    'final_map': LOSS_DICT[args.loss],
                    'residual_layer': LOSS_DICT['mae']}, 
                loss_weights={'final_map': W_MAP_VAR, 'residual_layer': W_RES_VAR},
                metrics={'final_map': ['mae', 'mse'],
                         'residual_layer': ['mae', 'mse']}
            )

    # --- 3. TRAINING ---
    logger.info("Inizio training...")
    
    callbacks = [
        LossWeightScheduler(W_RES_VAR, my_schedule),  # Aggiorna W_RES ad ogni epoca secondo la schedule definita
        # Salva solo il modello migliore sul validation set
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(models_dir, 'best_model.keras'),
            monitor='val_final_map_mae',
            mode='min',
            save_best_only=True,
            verbose=1
        ),
        # Ferma il training se non migliora 
        tf.keras.callbacks.EarlyStopping(
            monitor='val_final_map_mae',
            mode='min',
            patience=40,
            restore_best_weights=True,
            min_delta=1e-6,
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
    results = model.evaluate(test_ds)
    
    if args.residuals_only:
        logger.info(f"Test Loss Total: {results[0]:.6f}")
        logger.info(f"Test Loss Final Maps: {results[1]:.6f}")
        logger.info(f"Test Loss Residuals: {results[2]:.6f}")
    else:
        logger.info(f"Test Loss Total: {results[0]:.6f}")

    logger.info("Salvataggio di TUTTE le predizioni del test set...")
    
    all_inputs, all_truths, all_preds_final, all_preds_res = [], [], [], []

    # Iteriamo su tutto il dataset di test
    # verbose=1 per vedere il progresso se il test set è lungo
    for batch_x, batch_y in test_ds:
        # Predizione
        preds = model.predict(batch_x, verbose=0)
        
        all_inputs.append(batch_x.numpy())
        if args.residuals_only:
            # batch_y è un dizionario (perché abbiamo mappato prepare_residual_target)
            all_truths.append(batch_y['final_map'].numpy())
            all_preds_final.append(preds[0]) 
            all_preds_res.append(preds[1])   
        else:
            # batch_y è un tensore singolo
            all_truths.append(batch_y.numpy())
            all_preds_final.append(preds)

    # Concateniamo e salviamo
    logger.info("Concatenazione array...")
    final_inputs = np.concatenate(all_inputs, axis=0)
    final_truths = np.concatenate(all_truths, axis=0)
    final_preds = np.concatenate(all_preds_final, axis=0)

    np.save(os.path.join(args.output_dir, 'test_input.npy'), final_inputs)
    np.save(os.path.join(args.output_dir, 'test_truth.npy'), final_truths)
    np.save(os.path.join(args.output_dir, 'test_pred.npy'), final_preds)

    if args.residuals_only:
        final_res = np.concatenate(all_preds_res, axis=0)
        np.save(os.path.join(args.output_dir, 'test_residual.npy'), final_res)