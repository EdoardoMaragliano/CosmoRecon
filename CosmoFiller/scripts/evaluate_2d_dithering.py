#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# --- SETUP GPU (Identico al training per evitare OOM) ---
physical_gpus = tf.config.list_physical_devices('GPU')
if physical_gpus:
    try:
        # Usa solo la prima GPU disponibile per l'inferenza (solitamente sufficiente)
        tf.config.set_visible_devices(physical_gpus[0], 'GPU')
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"Warning GPU setup: {e}")

# --- IMPORTS ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from CosmoFiller.datahandler2d import create_split_datasets
    # Non importiamo il modello qui perché carichiamo direttamente il file compilato
except ImportError as e:
    print(f"ERRORE IMPORT: {e}")
    sys.exit(1)

# --- UTILS PLOTTING ---
def plot_results(inputs, targets, preds, save_path, n_samples=3):
    """
    Genera un plot comparativo: Input (Masked) | Target (Ground Truth) | Prediction | Residuals
    """
    rows = n_samples
    cols = 4 # Input, Target, Pred, Residual
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    
    for i in range(rows):
        # Gestione shape: (H, W, 1) -> (H, W) per plotting
        inp = inputs[i, :, :, 0]
        tar = targets[i, :, :, 0]
        pre = preds[i, :, :, 0]
        res = tar - pre # Residuo: Verità - Predizione

        # Determina min/max per scala colori coerente tra Target e Pred
        vmin = min(tar.min(), pre.min())
        vmax = max(tar.max(), pre.max())

        # Colonna 1: Input
        ax0 = axes[i, 0] if rows > 1 else axes[0]
        im0 = ax0.imshow(inp, cmap='inferno')
        ax0.set_title("Input (Masked)")
        plt.colorbar(im0, ax=ax0)

        # Colonna 2: Target
        ax1 = axes[i, 1] if rows > 1 else axes[1]
        im1 = ax1.imshow(tar, cmap='inferno', vmin=vmin, vmax=vmax)
        ax1.set_title("Target (Ground Truth)")
        plt.colorbar(im1, ax=ax1)

        # Colonna 3: Prediction
        ax2 = axes[i, 2] if rows > 1 else axes[2]
        im2 = ax2.imshow(pre, cmap='inferno', vmin=vmin, vmax=vmax)
        ax2.set_title("Inpainting Output")
        plt.colorbar(im2, ax=ax2)
        
        # Colonna 4: Residuals (Diverging colormap)
        ax3 = axes[i, 3] if rows > 1 else axes[3]
        im3 = ax3.imshow(res, cmap='seismic', vmin=-np.max(np.abs(res)), vmax=np.max(np.abs(res)))
        ax3.set_title("Residuals (Target - Pred)")
        plt.colorbar(im3, ax=ax3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot salvato in: {save_path}")

# -------------------------------
# MAIN INFERENCE
# -------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference 2D Inpainting")
    
    # Path Default (Modifica questi se vuoi default diversi)
    DEFAULT_H5 = '/home/emaragliano/Work/Projects/myfarm-disk/EuclidDithering/training_set_quijote_dithering_100mocks_5rot_1024pix_matched.h5'
    DEFAULT_MODEL_DIR = '/home/emaragliano/Work/Projects/myfarm-disk/AE_storage_Paper2/training_runs/dithering_inpainting_500epochs'
    OUTPUT_DIR = '/home/emaragliano/Work/Projects/myfarm-disk/AE_storage_Paper2/training_runs/dithering_inpainting_500_epochs_matched_dataset_1024pix'

    parser.add_argument('--model_path', type=str, default=os.path.join(DEFAULT_MODEL_DIR, 'models/best_model.keras'), help='Path del modello .keras')
    parser.add_argument('--h5_file', type=str, default=DEFAULT_H5, help='Path del file .h5')
    parser.add_argument('--output_dir', type=str, default=os.path.join(DEFAULT_MODEL_DIR, 'inference_results'), help='Dove salvare i risultati')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_samples', type=int, default=5, help='Numero di immagini da plottare')
    
    args = parser.parse_args()

    # Setup directories
    #os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"--- INIZIO INFERENZA ---")
    print(f"Modello: {args.model_path}")
    print(f"Dataset: {args.h5_file}")

    # 1. CARICAMENTO DATASET
    # Usiamo lo stesso loader per avere esattamente lo stesso split e preprocessing
    # Nota: scartiamo train e val, teniamo solo test_ds
    print("Caricamento Test Set...")
    _, _, test_ds = create_split_datasets(
        h5_path=args.h5_file,
        batch_size=args.batch_size,
        keys=('inputs', 'targets'), # Assumo che le chiavi siano standard, altrimenti aggiungi argomenti
        val_split=0.15,
        test_split=0.15,
        field_size=1024, # Assicurati che questo corrisponda al training
        dim=2,
        seed=42 # Importante: stesso seed del training per avere lo stesso test set!
    )

    # 2. CARICAMENTO MODELLO
    try:
        # Se nel training hai usato loss custom o layer custom, potresti dover passare 
        # custom_objects={'Huber': ...}. Solitamente Keras carica le loss standard automaticamente.
        model = tf.keras.models.load_model(args.model_path)
        print("Modello caricato con successo.")
    except Exception as e:
        print(f"Errore caricamento modello: {e}")
        sys.exit(1)

    # 3. ESECUZIONE SU UN BATCH DI TEST
    print("Esecuzione predizione su un batch...")
    
    # Preleviamo un batch dal dataset di test
    # iter() e next() prendono il primo batch disponibile nel generatore di test
    input_batch, target_batch = next(iter(test_ds))
    
    # Inferenza
    predictions = model.predict(input_batch)

    # 4. SALVATAGGIO DATI RAW (.npy)
    # Utile per fare analisi statistiche (spettri di potenza, istogrammi, ecc.) successivamente
    np.save(os.path.join(OUTPUT_DIR, 'inputs.npy'), input_batch.numpy())
    np.save(os.path.join(OUTPUT_DIR, 'targets.npy'), target_batch.numpy())
    np.save(os.path.join(OUTPUT_DIR, 'predictions.npy'), predictions)
    print(f"Array numpy salvati in {OUTPUT_DIR}")

    # 5. CALCOLO METRICHE RAPIDE
    mse = np.mean((target_batch.numpy() - predictions)**2)
    mae = np.mean(np.abs(target_batch.numpy() - predictions))
    print(f"Batch Metrics -> MSE: {mse:.5f}, MAE: {mae:.5f}")

    # 6. PLOTTING VISIVO
    print(f"Generazione plot per {args.n_samples} campioni...")
    plot_path = os.path.join(OUTPUT_DIR, 'visual_comparison.png')
    
    plot_results(
        input_batch.numpy(), 
        target_batch.numpy(), 
        predictions, 
        plot_path, 
        n_samples=min(args.n_samples, args.batch_size)
    )

    print("--- FINE ---")