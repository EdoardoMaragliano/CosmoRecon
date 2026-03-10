"""Tests for the SaveEveryNEpoch callback (D2)."""

import os
import warnings

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from CosmoRecon.checkpoints import SaveEveryNEpoch, EpochCheckpoint


class TestSaveEveryNEpoch:
    """Tests for ``SaveEveryNEpoch``."""

    @staticmethod
    def _build_tiny_model():
        inputs = keras.layers.Input(shape=(4,))
        outputs = keras.layers.Dense(1)(inputs)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="mse")
        return model

    def test_saves_at_interval(self, tmp_path):
        model = self._build_tiny_model()
        filepath = str(tmp_path / "model_{epoch:03d}.keras")
        cb = SaveEveryNEpoch(filepath=filepath, period=2)
        cb.set_model(model)

        for epoch in range(6):
            cb.on_epoch_end(epoch, logs={"loss": 1.0})

        # period=2 -> saves at epochs 2, 4, 6 (1-indexed)
        assert os.path.exists(tmp_path / "model_002.keras")
        assert os.path.exists(tmp_path / "model_004.keras")
        assert os.path.exists(tmp_path / "model_006.keras")
        assert not os.path.exists(tmp_path / "model_001.keras")
        assert not os.path.exists(tmp_path / "model_003.keras")

    def test_save_best_only_min(self, tmp_path):
        model = self._build_tiny_model()
        filepath = str(tmp_path / "model_{epoch:03d}.keras")
        cb = SaveEveryNEpoch(filepath=filepath, period=1,
                             monitor="val_loss", save_best_only=True, mode="min")
        cb.set_model(model)

        # Decreasing loss: epochs 1, 2, 3 should save; epoch 4 worsens
        losses = [1.0, 0.8, 0.5, 0.9]
        for epoch, loss in enumerate(losses):
            cb.on_epoch_end(epoch, logs={"val_loss": loss})

        assert os.path.exists(tmp_path / "model_001.keras")
        assert os.path.exists(tmp_path / "model_002.keras")
        assert os.path.exists(tmp_path / "model_003.keras")
        assert not os.path.exists(tmp_path / "model_004.keras")

    def test_save_best_only_max(self, tmp_path):
        model = self._build_tiny_model()
        filepath = str(tmp_path / "model_{epoch:03d}.keras")
        cb = SaveEveryNEpoch(filepath=filepath, period=1,
                             monitor="accuracy", save_best_only=True, mode="max")
        cb.set_model(model)

        accuracies = [0.5, 0.7, 0.6, 0.9]
        for epoch, acc in enumerate(accuracies):
            cb.on_epoch_end(epoch, logs={"accuracy": acc})

        assert os.path.exists(tmp_path / "model_001.keras")  # first is best
        assert os.path.exists(tmp_path / "model_002.keras")  # 0.7 > 0.5
        assert not os.path.exists(tmp_path / "model_003.keras")  # 0.6 < 0.7
        assert os.path.exists(tmp_path / "model_004.keras")  # 0.9 > 0.7

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            SaveEveryNEpoch(filepath="dummy", mode="bad")

    def test_monitor_missing_warns(self, tmp_path, caplog):
        model = self._build_tiny_model()
        filepath = str(tmp_path / "model_{epoch:03d}.keras")
        cb = SaveEveryNEpoch(filepath=filepath, period=1,
                             monitor="nonexistent", save_best_only=True)
        cb.set_model(model)

        import logging
        with caplog.at_level(logging.WARNING):
            cb.on_epoch_end(0, logs={"loss": 1.0})

        assert not os.path.exists(tmp_path / "model_001.keras")


class TestEpochCheckpointAlias:
    """Tests for the deprecated ``EpochCheckpoint`` alias."""

    def test_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cb = EpochCheckpoint(filepath="dummy", period=5)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "SaveEveryNEpoch" in str(w[0].message)
            assert isinstance(cb, SaveEveryNEpoch)
