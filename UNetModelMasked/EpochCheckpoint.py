from tensorflow import keras
import os

class EpochCheckpoint(keras.callbacks.Callback):
    def __init__(self, filepath, period=10):
        super().__init__()
        self.filepath = filepath
        self.period = period

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.period == 0:  # epoch parte da 0
            fname = self.filepath.format(epoch=epoch + 1)
            self.model.save(fname)
            print(f"\nSaved model at epoch {epoch+1} → {fname}")
