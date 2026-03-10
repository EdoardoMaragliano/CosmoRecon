"""Unit tests for train.py helper functions (A1–A4)."""

import argparse
import json
import logging
import random

import numpy as np
import pytest
import tensorflow as tf


# ---------------------------------------------------------------------------
# A1. _parse_gpu_indices
# ---------------------------------------------------------------------------

class TestParseGpuIndices:
    """Tests for ``_parse_gpu_indices``."""

    def test_none_returns_none(self, train_module):
        assert train_module._parse_gpu_indices(None) is None

    def test_single_index(self, train_module):
        assert train_module._parse_gpu_indices("0") == [0]

    def test_multiple_indices(self, train_module):
        assert train_module._parse_gpu_indices("0,1,3") == [0, 1, 3]

    def test_with_whitespace(self, train_module):
        assert train_module._parse_gpu_indices(" 0 , 2 ") == [0, 2]

    def test_empty_string(self, train_module):
        assert train_module._parse_gpu_indices("") == []


# ---------------------------------------------------------------------------
# A2. _load_json_params
# ---------------------------------------------------------------------------

class TestLoadJsonParams:
    """Tests for ``_load_json_params``."""

    @staticmethod
    def _make_args(**kwargs):
        """Create a minimal argparse Namespace with defaults."""
        defaults = {"epochs": 500, "batch_size": 16, "learning_rate": 1e-4}
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_known_keys_override(self, train_module, tmp_path):
        json_path = tmp_path / "p.json"
        json_path.write_text(json.dumps({"epochs": 10}))
        args = self._make_args()
        log = logging.getLogger("test")
        train_module._load_json_params(args, str(json_path), log)
        assert args.epochs == 10

    def test_multiple_overrides(self, train_module, tmp_path):
        json_path = tmp_path / "p.json"
        json_path.write_text(json.dumps({"epochs": 10, "batch_size": 4}))
        args = self._make_args()
        log = logging.getLogger("test")
        train_module._load_json_params(args, str(json_path), log)
        assert args.epochs == 10
        assert args.batch_size == 4

    def test_unknown_key_warns(self, train_module, tmp_path, caplog):
        json_path = tmp_path / "p.json"
        json_path.write_text(json.dumps({"bogus_key": 99, "epochs": 5}))
        args = self._make_args()
        log = logging.getLogger("test_warn")
        with caplog.at_level(logging.WARNING, logger="test_warn"):
            train_module._load_json_params(args, str(json_path), log)
        assert args.epochs == 5
        assert any("bogus_key" in r.message for r in caplog.records)

    def test_type_preservation(self, train_module, tmp_path):
        json_path = tmp_path / "p.json"
        json_path.write_text(json.dumps({"learning_rate": 0.001}))
        args = self._make_args()
        log = logging.getLogger("test")
        train_module._load_json_params(args, str(json_path), log)
        assert isinstance(args.learning_rate, float)
        assert args.learning_rate == pytest.approx(0.001)

    def test_empty_json(self, train_module, tmp_path):
        json_path = tmp_path / "p.json"
        json_path.write_text(json.dumps({}))
        args = self._make_args()
        log = logging.getLogger("test")
        train_module._load_json_params(args, str(json_path), log)
        assert args.epochs == 500  # unchanged


# ---------------------------------------------------------------------------
# A3. _set_random_seeds
# ---------------------------------------------------------------------------

class TestSetRandomSeeds:
    """Tests for ``_set_random_seeds``."""

    def test_determinism(self, train_module):
        train_module._set_random_seeds(42)
        np_val_1 = np.random.rand()
        py_val_1 = random.random()
        tf_val_1 = tf.random.uniform([1]).numpy()[0]

        train_module._set_random_seeds(42)
        np_val_2 = np.random.rand()
        py_val_2 = random.random()
        tf_val_2 = tf.random.uniform([1]).numpy()[0]

        assert np_val_1 == np_val_2
        assert py_val_1 == py_val_2
        assert tf_val_1 == pytest.approx(tf_val_2)


# ---------------------------------------------------------------------------
# A4. build_parser
# ---------------------------------------------------------------------------

class TestBuildParser:
    """Tests for ``build_parser``."""

    def test_defaults(self, train_module):
        parser = train_module.build_parser()
        args = parser.parse_args([])
        assert args.loss_type == "mse"
        assert args.batch_size == 16
        assert args.epochs == 500
        assert args.seed is None
        assert args.gpu_indices is None
        assert args.memory_growth is False

    def test_loss_type_choices(self, train_module):
        parser = train_module.build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--loss_type", "invalid"])

    def test_all_args(self, train_module):
        parser = train_module.build_parser()
        args = parser.parse_args([
            "--loss_type", "masked_gradient",
            "--obs_dir", "/obs",
            "--true_dir", "/true",
            "--mask_dir", "/masks",
            "--field_size", "64",
            "--batch_size", "8",
            "--epochs", "100",
            "--seed", "42",
            "--gpu_indices", "0,1",
            "--gradient_weight", "0.5",
            "--dropout",
            "--use_mask",
            "--memory_growth",
        ])
        assert args.loss_type == "masked_gradient"
        assert args.field_size == 64
        assert args.seed == 42
        assert args.gpu_indices == "0,1"
        assert args.gradient_weight == 0.5
        assert args.dropout is True
        assert args.use_mask is True
        assert args.memory_growth is True
