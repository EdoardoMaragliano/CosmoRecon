# checkpoints.py

"""
This module defines custom Keras callbacks for saving model checkpoints
at specified epoch intervals during training.
"""

from tensorflow import keras
import os
import numpy as np
import logging

from CosmoFiller.utils.loggers import setup_logger

# -------------------------
# Module logger
# -------------------------
logger = setup_logger(__name__)

# =====================
# Epoch Checkpoint Callback
# =====================
"""Custom Keras callback to save the model at specified epoch intervals."""

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

# -----------------------------
# Callback: save model every N epochs
# -----------------------------
class SaveEveryNEpoch(keras.callbacks.Callback):
    def __init__(self, filepath, period=10, monitor=None, save_best_only=False):
        super().__init__()
        self.period = int(period)
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.best = np.inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if (epoch + 1) % self.period == 0:
            # Optionally check monitor
            if self.save_best_only and self.monitor:
                current = logs.get(self.monitor)
                if current is None:
                    return
                if current < self.best:
                    self.best = current
                    fname = self.filepath.format(epoch=epoch + 1)
                    self.model.save(fname)
            elif not self.save_best_only:
                fname = self.filepath.format(epoch=epoch + 1)
                self.model.save(fname)