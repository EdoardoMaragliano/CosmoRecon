
import tensorflow as tf
from tensorflow.keras import layers, models

def build_alm_repair_net(input_shape, learning_rate=1e-4):
    """
    input_shape: (2 * numero_di_alm,) 
    Concatenazione di Re e Im degli alm fino a l_max
    """
    inputs = layers.Input(shape=input_shape)
    
    # --- Encoder / Feature Extraction ---
    x = layers.Dense(1024, activation='leaky_relu')(inputs)
    x = layers.BatchNormalization()(x)
    
    # --- Residual Blocks ---
    # Questi blocchi aiutano a raffinare i coefficienti senza perdere l'informazione originale
    for _ in range(3):
        res = x
        x = layers.Dense(1024, activation='leaky_relu')(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(1024, activation=None)(x)
        x = layers.Add()([x, res]) # Connessione residua
        x = layers.Activation('leaky_relu')(x)
        x = layers.BatchNormalization()(x)

    # --- Output Layer ---
    # L'output deve avere la stessa dimensione dell'input
    outputs = layers.Dense(input_shape[0], activation=None)(x)
    
    # Final Model: predice il target finale (o il residuo, a seconda di come configuri la loss)
    model = models.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model

# Esempio di utilizzo:
# Se l_max = 128, il numero di a_lm è (l_max + 1) * (l_max + 2) / 2 = 8385
# Poiché abbiamo Re e Im, l'input_shape sarà 16770.