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
'''
def create_split_datasets(h5_path, 
                          batch_size=8, 
                          keys=('x_data', 'y_data'), 
                          val_split=0.1, 
                          test_split=0.1,
                          field_size=128,
                          norm_val=40.0,
                          dim=2,
                          seed=42,
                          n_rotations=5):
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
    np.random.seed(seed)                # Per riproducibilità 
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
            lambda: h5_generator(h5_path, keys[0], keys[1], idx_list, norm_val, dim=dim),
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
    
    return train_ds, val_ds, test_ds'''

def create_split_datasets(h5_path, 
                          batch_size=8, 
                          keys=('x_data', 'y_data'), 
                          val_split=0.1, 
                          test_split=0.1,
                          field_size=128,
                          norm_val=40.0,
                          dim=2,
                          seed=42,
                          n_rotations=5): # <--- Aggiungiamo il numero di rotazioni
    """
    Crea dataset split assicurando che tutte le rotazioni di una singola 
    simulazione finiscano nello stesso split (niente leakage).
    """
    with h5py.File(h5_path, 'r') as f:
        n_samples = f[keys[0]].shape[0]
        
    # 1. Calcoliamo il numero di SIMULAZIONI originali
    n_simulations = n_samples // n_rotations
    print(f"Totale campioni: {n_samples} ({n_simulations} simulazioni con {n_rotations} rotazioni ciascuna)")

    # 2. Creiamo e mescoliamo gli indici delle SIMULAZIONI
    sim_indices = np.arange(n_simulations)
    np.random.seed(seed)
    np.random.shuffle(sim_indices)
    
    # 3. Calcoliamo i punti di taglio sulle simulazioni
    n_val_sim = int(n_simulations * val_split)
    n_test_sim = int(n_simulations * test_split)
    n_train_sim = n_simulations - n_val_sim - n_test_sim
    
    # 4. Funzione helper per espandere gli indici delle simulazioni in indici dei campioni
    def expand_indices(sim_list):
        expanded = []
        for sim_idx in sim_list:
            # Se la sim_idx è 0, prende i campioni 0,1,2,3,4
            # Se la sim_idx è 1, prende i campioni 5,6,7,8,9...
            start = sim_idx * n_rotations
            end = start + n_rotations
            expanded.extend(range(start, end))
        return np.array(expanded)

    train_indices = expand_indices(sim_indices[:n_train_sim])
    val_indices = expand_indices(sim_indices[n_train_sim : n_train_sim + n_val_sim])
    test_indices = expand_indices(sim_indices[n_train_sim + n_val_sim :])
    
    # Mischiamo di nuovo gli indici all'interno del solo training per non avere 
    # blocchi di rotazioni identiche consecutivi durante le epoche
    np.random.shuffle(train_indices)

    print(f"Split: {len(train_indices)} Train, {len(val_indices)} Val, {len(test_indices)} Test")

    # 5. Funzione helper per creare il dataset TF (rimane uguale a prima)
    def make_tf_dataset(idx_list):
        if dim == 2:
            shape = (field_size, field_size, 1)
        else:
            shape = (field_size, field_size, field_size, 1)
            
        ds = tf.data.Dataset.from_generator(
            lambda: h5_generator(h5_path, keys[0], keys[1], idx_list, norm_val, dim=dim),
            output_signature=(
                tf.TensorSpec(shape=shape, dtype=tf.float32),
                tf.TensorSpec(shape=shape, dtype=tf.float32)
            )
        )
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    return make_tf_dataset(train_indices), make_tf_dataset(val_indices), make_tf_dataset(test_indices)