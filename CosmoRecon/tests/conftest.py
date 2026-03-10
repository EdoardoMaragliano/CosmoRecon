"""Shared pytest fixtures for CosmoRecon tests."""

import importlib.util
import json
import logging
import os
import sys

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure the CosmoRecon package is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")

# Field size used across all tests — small enough for fast CPU execution
FIELD_SIZE = 16
N_FILES = 10
BASE_FILTERS = 4
MIN_SIZE = 4


# ---------------------------------------------------------------------------
# Script importers
# ---------------------------------------------------------------------------

def _import_script(script_name: str):
    """Import a Python script from ``scripts/`` as a module object."""
    path = os.path.join(SCRIPTS_DIR, script_name)
    module_name = script_name.replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def train_module():
    """Import ``scripts/train.py`` as a module."""
    return _import_script("train.py")


@pytest.fixture(scope="session")
def restart_module():
    """Import ``scripts/restart_training_from_saved_model.py`` as a module."""
    return _import_script("restart_training_from_saved_model.py")


@pytest.fixture(scope="session")
def evaluate_module():
    """Import ``scripts/evaluate_model.py`` as a module."""
    return _import_script("evaluate_model.py")


# ---------------------------------------------------------------------------
# Synthetic data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_data_dir(tmp_path):
    """Create ``obs/``, ``true/``, ``masks/`` directories with random ``.npy`` files.

    Returns a dict with paths and metadata.
    """
    obs_dir = tmp_path / "obs"
    true_dir = tmp_path / "true"
    mask_dir = tmp_path / "masks"
    obs_dir.mkdir()
    true_dir.mkdir()
    mask_dir.mkdir()

    rng = np.random.RandomState(42)
    for i in range(N_FILES):
        obs = rng.rand(FIELD_SIZE, FIELD_SIZE, FIELD_SIZE).astype(np.float32) * 20
        true = rng.rand(FIELD_SIZE, FIELD_SIZE, FIELD_SIZE).astype(np.float32) * 20
        np.save(obs_dir / f"mock_{i:03d}.npy", obs)
        np.save(true_dir / f"mock_{i:03d}.npy", true)

    # Binary mask: 1 = observed, 0 = missing (hole in the centre)
    mask = np.ones((FIELD_SIZE, FIELD_SIZE, FIELD_SIZE), dtype=np.float32)
    mask[4:12, 4:12, 4:12] = 0.0
    np.save(mask_dir / "mask.npy", mask)

    return {
        "root": tmp_path,
        "obs_dir": str(obs_dir),
        "true_dir": str(true_dir),
        "mask_dir": str(mask_dir),
        "n_files": N_FILES,
        "field_size": FIELD_SIZE,
    }


@pytest.fixture()
def params_json(tmp_path, synthetic_data_dir):
    """Create a ``params.json`` file pointing to the synthetic data."""
    params = {
        "obs_dir": synthetic_data_dir["obs_dir"],
        "true_dir": synthetic_data_dir["true_dir"],
        "mask_dir": synthetic_data_dir["mask_dir"],
        "output_dir": str(tmp_path / "output"),
        "field_size": FIELD_SIZE,
        "input_field": "rho",
        "base_filters": BASE_FILTERS,
        "batch_size": 2,
        "epochs": 2,
        "save_freq": 1,
        "learning_rate": 1e-3,
        "density_normalization": 20.0,
        "use_mask": True,
        "debug": False,
    }
    path = tmp_path / "params.json"
    path.write_text(json.dumps(params, indent=2))
    return str(path), params


# ---------------------------------------------------------------------------
# Helpers for running script main() functions
# ---------------------------------------------------------------------------

def run_script_main(main_fn, argv):
    """Call a script's ``main()`` with monkeypatched ``sys.argv``.

    Saves and restores ``sys.argv`` and cleans up root logging handlers
    so that tests do not leak state.
    """
    saved_argv = sys.argv[:]
    saved_handlers = logging.root.handlers[:]
    try:
        sys.argv = argv
        main_fn()
    finally:
        sys.argv = saved_argv
        # Restore logging state
        for h in logging.root.handlers[:]:
            if h not in saved_handlers:
                logging.root.removeHandler(h)
                h.close()


def common_train_args(synthetic_data_dir, output_dir, **overrides):
    """Build a list of CLI arguments for ``train.py``."""
    args = [
        "train.py",
        "--obs_dir", synthetic_data_dir["obs_dir"],
        "--true_dir", synthetic_data_dir["true_dir"],
        "--output_dir", str(output_dir),
        "--field_size", str(FIELD_SIZE),
        "--batch_size", "2",
        "--epochs", "2",
        "--save_freq", "1",
        "--base_filters", str(BASE_FILTERS),
        "--min_size", str(MIN_SIZE),
        "--learning_rate", "1e-3",
        "--density_normalization", "20.0",
    ]
    for key, value in overrides.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
        else:
            args.extend([flag, str(value)])
    return args
