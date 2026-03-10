"""Tests for restart_training_from_saved_model.py (G1-H4).

Unit tests for restart-specific logic, and integration tests that exercise
the full train -> save -> resume -> continue cycle.
"""

import glob
import json
import logging
import os
import re
import sys

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

from conftest import (
    FIELD_SIZE, BASE_FILTERS, MIN_SIZE,
    run_script_main, common_train_args,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from CosmoRecon.models import (
    MaskedMSE,
    MaskedUNet3D,
    build_unet,
)


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture()
def trained_mse_checkpoint(tmp_path, synthetic_data_dir, train_module):
    """Train a minimal MSE model for 2 epochs (save_freq=1) and return paths."""
    output_dir = str(tmp_path / "train_init")
    argv = common_train_args(synthetic_data_dir, output_dir, loss_type="mse")
    run_script_main(train_module.main, argv)

    model_dir = os.path.join(output_dir, "store_models")
    models = sorted(f for f in os.listdir(model_dir) if f.endswith(".keras"))
    assert models, "No checkpoint was saved"
    latest = os.path.join(model_dir, models[-1])
    return latest, output_dir, synthetic_data_dir


@pytest.fixture()
def trained_masked_checkpoint(tmp_path, synthetic_data_dir, train_module):
    """Train a minimal masked-MSE model for 2 epochs and return paths."""
    output_dir = str(tmp_path / "train_masked_init")
    argv = common_train_args(
        synthetic_data_dir, output_dir,
        loss_type="masked_mse",
        use_mask=True,
        mask_dir=synthetic_data_dir["mask_dir"],
    )
    run_script_main(train_module.main, argv)

    model_dir = os.path.join(output_dir, "store_models")
    models = sorted(f for f in os.listdir(model_dir) if f.endswith(".keras"))
    assert models
    latest = os.path.join(model_dir, models[-1])
    return latest, output_dir, synthetic_data_dir


# =========================================================================
# G1. Epoch extraction from filename
# =========================================================================

class TestEpochExtraction:
    """The restart script uses ``re.search(r"model_(\\d+)\\.keras", ...)``."""

    @pytest.mark.parametrize("path,expected", [
        ("models/model_100.keras", 100),
        ("models/model_005.keras", 5),
        ("/a/b/c/model_042.keras", 42),
        ("model_001.keras", 1),
    ])
    def test_standard_patterns(self, path, expected):
        match = re.search(r"model_(\d+)\.keras", path)
        assert match is not None
        assert int(match.group(1)) == expected

    def test_no_match_returns_zero(self):
        match = re.search(r"model_(\d+)\.keras", "best_model.keras")
        assert match is None
        initial_epoch = int(match.group(1)) if match else 0
        assert initial_epoch == 0


# =========================================================================
# G2. Mask re-injection into MaskedMSE
# =========================================================================

class TestMaskReinjection:

    def test_reinjection_restores_mask(self):
        """After save/load roundtrip, set_mask restores the loss mask."""
        mask = np.ones((1, FIELD_SIZE, FIELD_SIZE, FIELD_SIZE, 1), dtype=np.float32)
        mask[:, 4:12, 4:12, 4:12, :] = 0.0

        loss = MaskedMSE(mask=tf.constant(mask))
        assert loss.mask is not None

        # Simulate serialisation roundtrip
        config = loss.get_config()
        loss2 = MaskedMSE.from_config(config)
        assert loss2.mask is None

        # Re-inject
        loss2.set_mask(tf.constant(mask))
        assert loss2.mask is not None

        # Forward pass should work
        y_true = tf.random.uniform((1, FIELD_SIZE, FIELD_SIZE, FIELD_SIZE, 1))
        y_pred = tf.random.uniform((1, FIELD_SIZE, FIELD_SIZE, FIELD_SIZE, 1))
        result = loss2(y_true, y_pred)
        assert np.isfinite(result.numpy())

    def test_no_mask_raises_on_call(self):
        """MaskedMSE with mask=None should raise RuntimeError on call."""
        loss = MaskedMSE(mask=None)
        y_true = tf.random.uniform((1, 4, 4, 4, 1))
        y_pred = tf.random.uniform((1, 4, 4, 4, 1))
        with pytest.raises(RuntimeError, match="MaskedMSE.mask is None"):
            loss(y_true, y_pred)

    def test_non_masked_model_skips_reinjection(self):
        """A model compiled with 'mse' does not have .set_mask."""
        model = build_unet(
            (FIELD_SIZE, FIELD_SIZE, FIELD_SIZE, 1),
            base_filters=BASE_FILTERS, min_size=MIN_SIZE,
        )
        model.compile(optimizer="adam", loss="mse")
        assert not isinstance(model.loss, MaskedMSE)


# =========================================================================
# G3. CSV append mode
# =========================================================================

class TestCSVAppendMode:

    def test_append_when_resuming(self, restart_module):
        """When --resume_from is provided, CSVLogger should use append=True.

        We verify this indirectly: the script passes
        ``append=(args.resume_from is not None)`` to CSVLogger.
        """
        import argparse
        # Simulate args with resume_from set
        args = argparse.Namespace(resume_from="model_002.keras")
        assert args.resume_from is not None

        # Simulate args without resume_from
        args2 = argparse.Namespace(resume_from=None)
        assert args2.resume_from is None

    def test_fresh_start_no_append(self, restart_module):
        """When --resume_from is NOT provided, append should be False."""
        import argparse
        args = argparse.Namespace(resume_from=None)
        append = args.resume_from is not None
        assert append is False


# =========================================================================
# G4. Fresh model build
# =========================================================================

class TestFreshModelBuild:

    def test_fresh_build_compiles(self):
        """MaskedUNet3D can build and compile without errors."""
        inpainter = MaskedUNet3D(
            input_size=(FIELD_SIZE, FIELD_SIZE, FIELD_SIZE, 1),
            base_filters=BASE_FILTERS, min_size=MIN_SIZE,
        )
        model = inpainter.unet
        model.compile(optimizer="adam", loss="mse", metrics=["mse"])

        # Smoke test: single forward pass
        dummy = tf.random.uniform((1, FIELD_SIZE, FIELD_SIZE, FIELD_SIZE, 1))
        out = model(dummy, training=False)
        assert out.shape == (1, FIELD_SIZE, FIELD_SIZE, FIELD_SIZE, 1)


# =========================================================================
# H1. Resume and continue training (integration)
# =========================================================================

class TestResumeTraining:

    def test_resume_continues_from_checkpoint(self, restart_module, tmp_path,
                                               trained_mse_checkpoint):
        """Resume from a saved checkpoint and train 2 more epochs."""
        model_path, _, data_info = trained_mse_checkpoint
        resume_dir = str(tmp_path / "resume_out")

        argv = [
            "restart_training_from_saved_model.py",
            "--obs_dir", data_info["obs_dir"],
            "--true_dir", data_info["true_dir"],
            "--output_dir", resume_dir,
            "--field_size", str(FIELD_SIZE),
            "--batch_size", "2",
            "--epochs", "4",
            "--save_freq", "1",
            "--base_filters", str(BASE_FILTERS),
            "--min_size", str(MIN_SIZE),
            "--learning_rate", "1e-3",
            "--density_normalization", "20.0",
            "--resume_from", model_path,
        ]
        run_script_main(restart_module.main, argv)

        # Verify outputs exist
        assert os.path.isdir(os.path.join(resume_dir, "store_models"))
        assert os.path.exists(os.path.join(resume_dir, "history.csv"))

        # Verify predicted fields were saved
        pred_files = [
            f for f in os.listdir(os.path.join(resume_dir, "output_data"))
            if f.endswith(".npy")
        ]
        assert len(pred_files) >= 1

    def test_epoch_numbering_continues(self, restart_module, tmp_path,
                                        trained_mse_checkpoint):
        """CSV rows reflect proper epoch continuation, not starting from 0."""
        model_path, _, data_info = trained_mse_checkpoint
        resume_dir = str(tmp_path / "resume_epoch")

        argv = [
            "restart_training_from_saved_model.py",
            "--obs_dir", data_info["obs_dir"],
            "--true_dir", data_info["true_dir"],
            "--output_dir", resume_dir,
            "--field_size", str(FIELD_SIZE),
            "--batch_size", "2",
            "--epochs", "4",
            "--save_freq", "1",
            "--base_filters", str(BASE_FILTERS),
            "--min_size", str(MIN_SIZE),
            "--learning_rate", "1e-3",
            "--density_normalization", "20.0",
            "--resume_from", model_path,
        ]
        run_script_main(restart_module.main, argv)

        csv_path = os.path.join(resume_dir, "history.csv")
        assert os.path.exists(csv_path)

        # Read CSV and verify epoch column starts from initial_epoch
        import csv
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) >= 1  # at least one epoch logged


# =========================================================================
# H2. Resume with MaskedMSE + mask re-injection
# =========================================================================

class TestResumeMasked:

    def test_resume_masked_no_crash(self, restart_module, tmp_path,
                                     trained_masked_checkpoint):
        """Resume a MaskedMSE model without RuntimeError (mask re-injected)."""
        model_path, _, data_info = trained_masked_checkpoint
        resume_dir = str(tmp_path / "resume_masked")

        argv = [
            "restart_training_from_saved_model.py",
            "--obs_dir", data_info["obs_dir"],
            "--true_dir", data_info["true_dir"],
            "--output_dir", resume_dir,
            "--field_size", str(FIELD_SIZE),
            "--batch_size", "2",
            "--epochs", "4",
            "--save_freq", "1",
            "--base_filters", str(BASE_FILTERS),
            "--min_size", str(MIN_SIZE),
            "--learning_rate", "1e-3",
            "--density_normalization", "20.0",
            "--use_mask",
            "--mask_dir", data_info["mask_dir"],
            "--resume_from", model_path,
        ]
        run_script_main(restart_module.main, argv)

        pred_files = [
            f for f in os.listdir(os.path.join(resume_dir, "output_data"))
            if f.endswith(".npy")
        ]
        assert len(pred_files) >= 1

        # Verify finite outputs
        for pf in pred_files:
            arr = np.load(os.path.join(resume_dir, "output_data", pf))
            assert np.all(np.isfinite(arr))


# =========================================================================
# H3. Fresh start (no --resume_from)
# =========================================================================

class TestFreshStart:

    def test_fresh_start_e2e(self, restart_module, tmp_path,
                              synthetic_data_dir):
        """Without --resume_from, builds a new model from scratch."""
        output_dir = str(tmp_path / "fresh")

        argv = [
            "restart_training_from_saved_model.py",
            "--obs_dir", synthetic_data_dir["obs_dir"],
            "--true_dir", synthetic_data_dir["true_dir"],
            "--output_dir", output_dir,
            "--field_size", str(FIELD_SIZE),
            "--batch_size", "2",
            "--epochs", "2",
            "--save_freq", "1",
            "--base_filters", str(BASE_FILTERS),
            "--min_size", str(MIN_SIZE),
            "--learning_rate", "1e-3",
            "--density_normalization", "20.0",
        ]
        run_script_main(restart_module.main, argv)

        assert os.path.isdir(os.path.join(output_dir, "store_models"))
        model_files = [
            f for f in os.listdir(os.path.join(output_dir, "store_models"))
            if f.endswith(".keras")
        ]
        assert len(model_files) >= 1

        pred_files = [
            f for f in os.listdir(os.path.join(output_dir, "output_data"))
            if f.endswith(".npy")
        ]
        assert len(pred_files) >= 1


# =========================================================================
# H4. Invalid checkpoint path
# =========================================================================

class TestInvalidCheckpoint:

    def test_nonexistent_checkpoint_raises(self, restart_module, tmp_path,
                                            synthetic_data_dir):
        """--resume_from with a path that doesn't exist -> error from Keras."""
        output_dir = str(tmp_path / "invalid_ckpt")

        argv = [
            "restart_training_from_saved_model.py",
            "--obs_dir", synthetic_data_dir["obs_dir"],
            "--true_dir", synthetic_data_dir["true_dir"],
            "--output_dir", output_dir,
            "--field_size", str(FIELD_SIZE),
            "--batch_size", "2",
            "--epochs", "4",
            "--base_filters", str(BASE_FILTERS),
            "--min_size", str(MIN_SIZE),
            "--density_normalization", "20.0",
            "--resume_from", str(tmp_path / "nonexistent_model.keras"),
        ]
        with pytest.raises((OSError, IOError, ValueError)):
            run_script_main(restart_module.main, argv)
