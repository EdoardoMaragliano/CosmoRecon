import tensorflow as tf
import h5py
import numpy as np

def h5_generator(h5_path, x_key, y_key, indices, norm_val=40.0, dim=2):
    """Generatore che legge solo gli indici specifici passati."""
    with h5py.File(h5_path, 'r') as f:
        ds_x = f[x_key]
        ds_y = f[y_key]
        
        for i in indices:
            x_sample = ds_x[i]
            y_sample = ds_y[i]
            
            # Pre-processing
            x_sample = x_sample.astype(np.float32) / norm_val
            y_sample = y_sample.astype(np.float32) / norm_val
            
            # Aggiunta canale se necessario
            if x_sample.ndim == dim: 
                x_sample = np.expand_dims(x_sample, axis=-1)
                y_sample = np.expand_dims(y_sample, axis=-1)
                
            yield x_sample, y_sample

def create_split_datasets(h5_path, 
                          batch_size=8, 
                          keys=('x_data', 'y_data'), 
                          val_split=0.1, 
                          test_split=0.1,
                          field_size=128,
                          dim=2,
                          seed=42):
    """
    Crea 3 dataset separati (Train, Val, Test) dallo stesso file H5
    senza caricare i dati in memoria.
    """
    
    # 1. Leggiamo il numero totale di campioni
    with h5py.File(h5_path, 'r') as f:
        n_samples = f[keys[0]].shape[0]
        
    print(f"Totale campioni nel file: {n_samples}")

    # 2. Creiamo e mescoliamo gli indici
    indices = np.arange(n_samples)
    np.random.seed(seed) # Per riproducibilità (importante in scienza!)
    np.random.shuffle(indices)
    
    # 3. Calcoliamo i punti di taglio
    n_val = int(n_samples * val_split)
    n_test = int(n_samples * test_split)
    n_train = n_samples - n_val - n_test
    
    # 4. Dividiamo gli indici
    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val :]
    
    print(f"Split: {len(train_indices)} Train, {len(val_indices)} Val, {len(test_indices)} Test")

    # 5. Funzione helper per creare il dataset TF da una lista di indici
    def make_tf_dataset(idx_list):
        if dim == 2:
            shape = (field_size, field_size, 1)
        else:
            shape = (field_size, field_size, field_size, 1)
            
        ds = tf.data.Dataset.from_generator(
            lambda: h5_generator(h5_path, keys[0], keys[1], idx_list, dim=dim),
            output_signature=(
                tf.TensorSpec(shape=shape, dtype=tf.float32),
                tf.TensorSpec(shape=shape, dtype=tf.float32)
            )
        )
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    # 6. Creazione dei 3 oggetti Dataset
    train_ds = make_tf_dataset(train_indices)
    val_ds = make_tf_dataset(val_indices)
    test_ds = make_tf_dataset(test_indices)
    
    return train_ds, val_ds, test_ds