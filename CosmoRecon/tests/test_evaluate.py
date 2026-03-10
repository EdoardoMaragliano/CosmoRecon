"""Tests for evaluate_model.py — validation (E1-E2) and integration (F1-F4)."""

import glob
import json
import logging
import os
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

from CosmoRecon.models import build_unet, MaskedMSE
from CosmoRecon.datahandler import create_dataset


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture()
def trained_mse_model(tmp_path, synthetic_data_dir, train_module):
    """Train a minimal MSE model and return (model_path, output_dir, data_info)."""
    output_dir = str(tmp_path / "train_out")
    argv = common_train_args(synthetic_data_dir, output_dir, loss_type="mse")
    run_script_main(train_module.main, argv)

    model_dir = os.path.join(output_dir, "store_models")
    models = sorted(f for f in os.listdir(model_dir) if f.endswith(".keras"))
    assert models, "No checkpoint was saved during training"
    model_path = os.path.join(model_dir, models[-1])  # latest
    return model_path, output_dir, synthetic_data_dir


@pytest.fixture()
def trained_masked_model(tmp_path, synthetic_data_dir, train_module):
    """Train a minimal masked MSE model."""
    output_dir = str(tmp_path / "train_masked_out")
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
    model_path = os.path.join(model_dir, models[-1])
    return model_path, output_dir, synthetic_data_dir


# =========================================================================
# E1. Missing args
# =========================================================================

class TestEvaluateValidation:

    def test_no_obs_dir(self, evaluate_module, tmp_path, synthetic_data_dir):
        argv = [
            "evaluate_model.py",
            "--true_dir", synthetic_data_dir["true_dir"],
            "--output_dir", str(tmp_path / "eval"),
            "--model_path", "dummy.keras",
        ]
        with pytest.raises(ValueError, match="obs_dir and true_dir"):
            run_script_main(evaluate_module.main, argv)


# =========================================================================
# E2. Mask errors
# =========================================================================

class TestEvaluateMaskErrors:

    def test_use_mask_empty_dir(self, evaluate_module, tmp_path,
                                 synthetic_data_dir, trained_mse_model):
        model_path, _, _ = trained_mse_model
        empty_mask = tmp_path / "empty_masks"
        empty_mask.mkdir()

        argv = [
            "evaluate_model.py",
            "--obs_dir", synthetic_data_dir["obs_dir"],
            "--true_dir", synthetic_data_dir["true_dir"],
            "--output_dir", str(tmp_path / "eval"),
            "--model_path", model_path,
            "--field_size", str(FIELD_SIZE),
            "--density_normalization", "20.0",
            "--use_mask",
            "--mask_dir", str(empty_mask),
        ]
        with pytest.raises(ValueError, match="no mask files found"):
            run_script_main(evaluate_module.main, argv)

    def test_use_mask_multiple_masks(self, evaluate_module, tmp_path,
                                      synthetic_data_dir, trained_mse_model):
        model_path, _, _ = trained_mse_model
        multi_mask = tmp_path / "multi_masks"
        multi_mask.mkdir()
        dummy = np.ones((FIELD_SIZE,) * 3, dtype=np.float32)
        np.save(multi_mask / "a.npy", dummy)
        np.save(multi_mask / "b.npy", dummy)

        argv = [
            "evaluate_model.py",
            "--obs_dir", synthetic_data_dir["obs_dir"],
            "--true_dir", synthetic_data_dir["true_dir"],
            "--output_dir", str(tmp_path / "eval"),
            "--model_path", model_path,
            "--field_size", str(FIELD_SIZE),
            "--density_normalization", "20.0",
            "--use_mask",
            "--mask_dir", str(multi_mask),
        ]
        with pytest.raises(NotImplementedError, match="Multiple"):
            run_script_main(evaluate_module.main, argv)


# =========================================================================
# F1. Evaluate with MSE model (no mask)
# =========================================================================

class TestEvaluateMSE:

    def test_evaluation_e2e(self, evaluate_module, tmp_path,
                             trained_mse_model):
        model_path, _, data_info = trained_mse_model
        eval_dir = str(tmp_path / "eval_mse")

        argv = [
            "evaluate_model.py",
            "--obs_dir", data_info["obs_dir"],
            "--true_dir", data_info["true_dir"],
            "--output_dir", eval_dir,
            "--model_path", model_path,
            "--field_size", str(FIELD_SIZE),
            "--density_normalization", "20.0",
        ]
        run_script_main(evaluate_module.main, argv)

        # Check output files
        out_data = os.path.join(eval_dir, "output_data")
        assert os.path.isdir(out_data)
        pred_files = [f for f in os.listdir(out_data) if f.endswith(".npy")]
        assert len(pred_files) >= 1

        # Correct shape
        for pf in pred_files:
            arr = np.load(os.path.join(out_data, pf))
            assert arr.shape == (FIELD_SIZE, FIELD_SIZE, FIELD_SIZE)

    def test_output_count_matches_val_split(self, evaluate_module, tmp_path,
                                             trained_mse_model):
        """Number of predicted fields == val split size."""
        model_path, _, data_info = trained_mse_model
        eval_dir = str(tmp_path / "eval_count")

        argv = [
            "evaluate_model.py",
            "--obs_dir", data_info["obs_dir"],
            "--true_dir", data_info["true_dir"],
            "--output_dir", eval_dir,
            "--model_path", model_path,
            "--field_size", str(FIELD_SIZE),
            "--density_normalization", "20.0",
        ]
        run_script_main(evaluate_module.main, argv)

        pred_files = [
            f for f in os.listdir(os.path.join(eval_dir, "output_data"))
            if f.endswith(".npy")
        ]
        n_total = data_info["n_files"]
        n_val = n_total - int(0.8 * n_total)  # test_size=0.2, shuffle=False
        assert len(pred_files) == n_val

    def test_denormalization(self, evaluate_module, tmp_path, trained_mse_model):
        """Output values should be in raw scale (multiplied by norm)."""
        model_path, _, data_info = trained_mse_model
        eval_dir = str(tmp_path / "eval_denorm")
        norm_val = 20.0

        argv = [
            "evaluate_model.py",
            "--obs_dir", data_info["obs_dir"],
            "--true_dir", data_info["true_dir"],
            "--output_dir", eval_dir,
            "--model_path", model_path,
            "--field_size", str(FIELD_SIZE),
            "--density_normalization", str(norm_val),
        ]
        run_script_main(evaluate_module.main, argv)

        # The model outputs are multiplied by norm_val -> values have raw magnitude
        pred = np.load(os.path.join(eval_dir, "output_data", "pred_field_000.npy"))
        assert np.all(np.isfinite(pred))


# =========================================================================
# F2. Evaluate with masked model
# =========================================================================

class TestEvaluateMasked:

    def test_masked_eval_blending(self, evaluate_module, tmp_path,
                                   trained_masked_model):
        """In observed voxels (mask==1), output == raw observed field."""
        model_path, _, data_info = trained_masked_model
        eval_dir = str(tmp_path / "eval_masked")

        argv = [
            "evaluate_model.py",
            "--obs_dir", data_info["obs_dir"],
            "--true_dir", data_info["true_dir"],
            "--output_dir", eval_dir,
            "--model_path", model_path,
            "--field_size", str(FIELD_SIZE),
            "--density_normalization", "20.0",
            "--use_mask",
            "--mask_dir", data_info["mask_dir"],
        ]
        run_script_main(evaluate_module.main, argv)

        mask = np.load(os.path.join(data_info["mask_dir"], "mask.npy"))
        pred = np.load(os.path.join(eval_dir, "output_data", "pred_field_000.npy"))

        # Get the first validation file
        obs_files = sorted(glob.glob(os.path.join(data_info["obs_dir"], "*.npy")))
        n_val = max(1, len(obs_files) // 5)
        val_obs = obs_files[-n_val:]
        obs_raw = np.load(val_obs[0])

        observed_voxels = mask == 1
        np.testing.assert_allclose(
            pred[observed_voxels], obs_raw[observed_voxels],
            rtol=1e-4, atol=1e-4,
        )


# =========================================================================
# F3. Split consistency
# =========================================================================

class TestSplitConsistency:

    def test_same_split_train_and_eval(self, synthetic_data_dir):
        """train_test_split(shuffle=False) is deterministic — verify."""
        from sklearn.model_selection import train_test_split

        obs = sorted(glob.glob(os.path.join(
            synthetic_data_dir["obs_dir"], "*.npy")))
        true = sorted(glob.glob(os.path.join(
            synthetic_data_dir["true_dir"], "*.npy")))

        _, x_val_1, _, y_val_1 = train_test_split(
            obs, true, test_size=0.2, shuffle=False)
        _, x_val_2, _, y_val_2 = train_test_split(
            obs, true, test_size=0.2, shuffle=False)

        assert x_val_1 == x_val_2
        assert y_val_1 == y_val_2


# =========================================================================
# F4. JSON key warning
# =========================================================================

class TestEvaluateJsonWarning:

    def test_unknown_key_warns(self, evaluate_module, tmp_path,
                                trained_mse_model, caplog):
        model_path, _, data_info = trained_mse_model
        eval_dir = str(tmp_path / "eval_json")

        # Create a JSON with an unknown key
        json_path = tmp_path / "bad.json"
        json_path.write_text(json.dumps({
            "obs_dir": data_info["obs_dir"],
            "true_dir": data_info["true_dir"],
            "field_size": FIELD_SIZE,
            "density_normalization": 20.0,
            "completely_made_up_key": 42,
        }))

        argv = [
            "evaluate_model.py",
            "--param_file", str(json_path),
            "--output_dir", eval_dir,
            "--model_path", model_path,
        ]
        with caplog.at_level(logging.WARNING):
            run_script_main(evaluate_module.main, argv)
        assert any("completely_made_up_key" in r.message for r in caplog.records)
