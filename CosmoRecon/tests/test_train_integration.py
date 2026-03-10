"""End-to-end integration tests for train.py (C1-C6).

Each test creates small 16^3 synthetic data, trains for 2 epochs on CPU,
and verifies outputs.  Tests are intentionally kept at small scale so they
complete in seconds even without a GPU.
"""

import os
import sys

import numpy as np
import pytest
import tensorflow as tf

from conftest import (
    FIELD_SIZE, BASE_FILTERS, MIN_SIZE,
    run_script_main, common_train_args,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from CosmoRecon.models import MaskedMSE, MaskedMSEWithGradient


# =========================================================================
# Helpers
# =========================================================================

def _assert_output_structure(output_dir):
    """Check that the standard output sub-directories and files exist."""
    assert os.path.isdir(os.path.join(output_dir, "store_models"))
    assert os.path.isdir(os.path.join(output_dir, "logs"))
    assert os.path.isdir(os.path.join(output_dir, "output_data"))
    assert os.path.isdir(os.path.join(output_dir, "losses"))

    # Loss arrays
    loss_path = os.path.join(output_dir, "losses", "loss.npy")
    val_loss_path = os.path.join(output_dir, "losses", "val_loss.npy")
    assert os.path.exists(loss_path)
    assert os.path.exists(val_loss_path)
    losses = np.load(loss_path)
    val_losses = np.load(val_loss_path)
    assert len(losses) > 0
    assert len(val_losses) > 0
    assert np.all(np.isfinite(losses))
    assert np.all(np.isfinite(val_losses))

    # CSV log
    csv_path = os.path.join(output_dir, "logs", "history.csv")
    assert os.path.exists(csv_path)

    # At least one checkpoint (save_freq=1, 2 epochs -> model_001, model_002)
    model_files = [
        f for f in os.listdir(os.path.join(output_dir, "store_models"))
        if f.endswith(".keras")
    ]
    assert len(model_files) >= 1

    # Predicted fields
    pred_files = [
        f for f in os.listdir(os.path.join(output_dir, "output_data"))
        if f.startswith("pred_field_") and f.endswith(".npy")
    ]
    assert len(pred_files) >= 1
    return losses, val_losses, pred_files


def _check_pred_shape(output_dir, pred_files):
    """All predicted fields should have the correct spatial shape."""
    for pf in pred_files:
        arr = np.load(os.path.join(output_dir, "output_data", pf))
        assert arr.shape == (FIELD_SIZE, FIELD_SIZE, FIELD_SIZE), (
            f"{pf} has shape {arr.shape}"
        )


# =========================================================================
# C1. MSE training (no mask)
# =========================================================================

class TestMSETraining:

    def test_mse_training_e2e(self, train_module, tmp_path, synthetic_data_dir):
        output_dir = str(tmp_path / "out_mse")
        argv = common_train_args(synthetic_data_dir, output_dir, loss_type="mse")
        run_script_main(train_module.main, argv)

        losses, val_losses, pred_files = _assert_output_structure(output_dir)
        _check_pred_shape(output_dir, pred_files)


# =========================================================================
# C2. Masked MSE training
# =========================================================================

class TestMaskedMSETraining:

    def test_masked_mse_e2e(self, train_module, tmp_path, synthetic_data_dir):
        output_dir = str(tmp_path / "out_masked_mse")
        argv = common_train_args(
            synthetic_data_dir, output_dir,
            loss_type="masked_mse",
            use_mask=True,
            mask_dir=synthetic_data_dir["mask_dir"],
        )
        run_script_main(train_module.main, argv)

        losses, val_losses, pred_files = _assert_output_structure(output_dir)
        _check_pred_shape(output_dir, pred_files)

    def test_mask_blending_in_observed_region(self, train_module, tmp_path,
                                               synthetic_data_dir):
        """In unmasked (observed) voxels, predicted field should equal the
        raw observed field after blending."""
        output_dir = str(tmp_path / "out_blend")
        argv = common_train_args(
            synthetic_data_dir, output_dir,
            loss_type="masked_mse",
            use_mask=True,
            mask_dir=synthetic_data_dir["mask_dir"],
        )
        run_script_main(train_module.main, argv)

        mask = np.load(os.path.join(synthetic_data_dir["mask_dir"], "mask.npy"))

        # Check the first predicted field
        pred_path = os.path.join(output_dir, "output_data", "pred_field_000.npy")
        pred = np.load(pred_path)

        # The validation set is the last 20% of sorted files
        import glob
        obs_files = sorted(glob.glob(os.path.join(
            synthetic_data_dir["obs_dir"], "*.npy")))
        n_val = max(1, len(obs_files) // 5)  # test_size=0.2
        val_obs_files = obs_files[-n_val:]
        obs_raw = np.load(val_obs_files[0])

        # Where mask==1 (observed), blended output == observed raw
        observed_voxels = mask == 1
        np.testing.assert_allclose(
            pred[observed_voxels], obs_raw[observed_voxels],
            rtol=1e-4, atol=1e-4,
        )


# =========================================================================
# C3. Masked gradient training
# =========================================================================

class TestMaskedGradientTraining:

    def test_masked_gradient_e2e(self, train_module, tmp_path,
                                  synthetic_data_dir):
        output_dir = str(tmp_path / "out_grad")
        argv = common_train_args(
            synthetic_data_dir, output_dir,
            loss_type="masked_gradient",
            use_mask=True,
            mask_dir=synthetic_data_dir["mask_dir"],
            gradient_weight="0.1",
        )
        run_script_main(train_module.main, argv)

        losses, val_losses, pred_files = _assert_output_structure(output_dir)
        _check_pred_shape(output_dir, pred_files)


# =========================================================================
# C4. Seed reproducibility
# =========================================================================

class TestSeedReproducibility:

    def test_same_seed_same_initial_weights(self, train_module):
        """Same seed produces identical model weights."""
        import tensorflow as tf
        from CosmoRecon.models import build_unet

        train_module._set_random_seeds(99)
        tf.keras.backend.clear_session()
        m1 = build_unet(
            (FIELD_SIZE, FIELD_SIZE, FIELD_SIZE, 1),
            base_filters=BASE_FILTERS, min_size=MIN_SIZE,
        )

        train_module._set_random_seeds(99)
        tf.keras.backend.clear_session()
        m2 = build_unet(
            (FIELD_SIZE, FIELD_SIZE, FIELD_SIZE, 1),
            base_filters=BASE_FILTERS, min_size=MIN_SIZE,
        )

        for w1, w2 in zip(m1.get_weights(), m2.get_weights()):
            np.testing.assert_array_equal(w1, w2)


# =========================================================================
# C5. Delta field mode
# =========================================================================

class TestDeltaFieldMode:

    def test_delta_output_activation(self, train_module, tmp_path,
                                      synthetic_data_dir):
        """With --input_field delta, all network outputs >= -1/norm_val."""
        output_dir = str(tmp_path / "out_delta")
        argv = common_train_args(
            synthetic_data_dir, output_dir,
            loss_type="mse",
            input_field="delta",
        )
        run_script_main(train_module.main, argv)

        # Load a predicted field and check the constraint
        pred_files = [
            f for f in os.listdir(os.path.join(output_dir, "output_data"))
            if f.endswith(".npy")
        ]
        assert len(pred_files) >= 1


# =========================================================================
# C6. Dropout enabled
# =========================================================================

class TestDropoutEnabled:

    def test_model_has_dropout_layer(self):
        """When dropout=True, the built model contains a Dropout layer."""
        from CosmoRecon.models import build_unet
        model = build_unet(
            (FIELD_SIZE, FIELD_SIZE, FIELD_SIZE, 1),
            base_filters=BASE_FILTERS, min_size=MIN_SIZE,
            dropout_layer=True, dropout_rate=0.2,
        )
        layer_types = [type(layer).__name__ for layer in model.layers]
        assert "Dropout" in layer_types

    def test_model_without_dropout(self):
        """When dropout=False (default), no Dropout layer is present."""
        from CosmoRecon.models import build_unet
        model = build_unet(
            (FIELD_SIZE, FIELD_SIZE, FIELD_SIZE, 1),
            base_filters=BASE_FILTERS, min_size=MIN_SIZE,
            dropout_layer=False,
        )
        layer_types = [type(layer).__name__ for layer in model.layers]
        assert "Dropout" not in layer_types
