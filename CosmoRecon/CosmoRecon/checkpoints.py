"""
Custom Keras callbacks for periodic model checkpointing.

Provides ``SaveEveryNEpoch``, a callback that saves the model at fixed epoch
intervals with optional best-metric tracking.  ``EpochCheckpoint`` is kept as
a convenience alias.
"""

import numpy as np
import warnings
from typing import Dict, Optional
from tensorflow import keras

from CosmoRecon.utils.loggers import setup_logger

logger = setup_logger(__name__)


class SaveEveryNEpoch(keras.callbacks.Callback):
    """Save the model every *period* epochs, optionally only when a monitored
    metric improves.

    Parameters
    ----------
    filepath : str
        Path template with an ``{epoch}`` placeholder,
        e.g. ``"models/model_{epoch:03d}.keras"``.
    period : int
        Save frequency in epochs.
    monitor : str or None
        Metric name to watch.  If ``None``, the model is saved unconditionally
        every *period* epochs.
    save_best_only : bool
        When ``True`` (and *monitor* is set), the model is saved only if the
        monitored metric improves.
    mode : {'min', 'max'}
        ``'min'`` means improvement = metric decrease (e.g. loss);
        ``'max'`` means improvement = metric increase (e.g. accuracy).
    """

    def __init__(self, filepath: str, period: int = 10,
                 monitor: Optional[str] = None,
                 save_best_only: bool = False, mode: str = 'min'):
        super().__init__()
        self.filepath = filepath
        self.period = int(period)
        self.monitor = monitor
        self.save_best_only = save_best_only

        if mode not in ('min', 'max'):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")
        self.mode = mode
        self.best = np.inf if mode == 'min' else -np.inf

    def _is_improvement(self, current: float) -> bool:
        if self.mode == 'min':
            return current < self.best
        return current > self.best

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        logs = logs or {}
        if (epoch + 1) % self.period != 0:
            return

        if self.save_best_only and self.monitor:
            current = logs.get(self.monitor)
            if current is None:
                logger.warning(
                    "SaveEveryNEpoch: monitor '%s' not found in logs at epoch %d",
                    self.monitor, epoch + 1,
                )
                return
            if self._is_improvement(current):
                self.best = current
                fname = self.filepath.format(epoch=epoch + 1)
                self.model.save(fname)
                logger.info("Saved best model at epoch %d -> %s", epoch + 1, fname)
        else:
            fname = self.filepath.format(epoch=epoch + 1)
            self.model.save(fname)
            logger.info("Saved model at epoch %d -> %s", epoch + 1, fname)


def EpochCheckpoint(*args, **kwargs):
    """Deprecated alias for :class:`SaveEveryNEpoch`."""
    warnings.warn(
        "EpochCheckpoint is deprecated, use SaveEveryNEpoch instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return SaveEveryNEpoch(*args, **kwargs)
