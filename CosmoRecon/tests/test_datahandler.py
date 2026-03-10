"""Tests for the data handler module (D1)."""

import numpy as np
import pytest
import tensorflow as tf

from conftest import FIELD_SIZE

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from CosmoRecon.datahandler import create_dataset


class TestCreateDataset:
    """Tests for ``create_dataset``."""

    @pytest.fixture()
    def npy_files(self, tmp_path):
        """Create a small set of .npy file pairs and a mask."""
        n = 5
        rng = np.random.RandomState(0)
        obs_files, true_files = [], []
        for i in range(n):
            obs = rng.rand(FIELD_SIZE, FIELD_SIZE, FIELD_SIZE).astype(np.float32) * 20
            true = rng.rand(FIELD_SIZE, FIELD_SIZE, FIELD_SIZE).astype(np.float32) * 20
            obs_path = str(tmp_path / f"obs_{i}.npy")
            true_path = str(tmp_path / f"true_{i}.npy")
            np.save(obs_path, obs)
            np.save(true_path, true)
            obs_files.append(obs_path)
            true_files.append(true_path)

        mask = np.ones((FIELD_SIZE, FIELD_SIZE, FIELD_SIZE), dtype=np.float32)
        mask[4:12, 4:12, 4:12] = 0.0
        return obs_files, true_files, mask

    def test_basic_shapes(self, npy_files):
        obs_files, true_files, _ = npy_files
        ds = create_dataset(obs_files, true_files, batch_size=2,
                            field_size=FIELD_SIZE, norm_val=20.0, shuffle=False)
        for x, y in ds.take(1):
            assert x.shape == (2, FIELD_SIZE, FIELD_SIZE, FIELD_SIZE, 1)
            assert y.shape == (2, FIELD_SIZE, FIELD_SIZE, FIELD_SIZE, 1)

    def test_normalization(self, npy_files):
        obs_files, true_files, _ = npy_files
        ds = create_dataset(obs_files, true_files, batch_size=1,
                            field_size=FIELD_SIZE, norm_val=20.0, shuffle=False)
        for x, y in ds.take(1):
            # raw values are in [0, 20], so normalised should be in [0, 1]
            assert float(tf.reduce_max(x)) <= 1.0 + 1e-5
            assert float(tf.reduce_min(x)) >= -1e-5
            break

    def test_two_channel_with_mask(self, npy_files):
        obs_files, true_files, mask = npy_files
        ds = create_dataset(obs_files, true_files, batch_size=2, channels=2,
                            field_size=FIELD_SIZE, norm_val=20.0, mask=mask,
                            shuffle=False)
        for x, y in ds.take(1):
            assert x.shape[-1] == 2  # obs*mask + mask
            assert y.shape[-1] == 1
            break

    def test_two_channel_without_mask_raises(self, npy_files):
        obs_files, true_files, _ = npy_files
        with pytest.raises(ValueError, match="channels=2 requires a mask"):
            create_dataset(obs_files, true_files, batch_size=2, channels=2,
                           field_size=FIELD_SIZE, norm_val=20.0, mask=None)

    def test_invalid_channels_raises(self, npy_files):
        obs_files, true_files, _ = npy_files
        with pytest.raises(ValueError, match="channels must be 1 or 2"):
            create_dataset(obs_files, true_files, batch_size=2, channels=3,
                           field_size=FIELD_SIZE, norm_val=20.0)

    def test_single_channel_with_mask(self, npy_files):
        obs_files, true_files, mask = npy_files
        ds = create_dataset(obs_files, true_files, batch_size=2, channels=1,
                            field_size=FIELD_SIZE, norm_val=20.0, mask=mask,
                            shuffle=False)
        for x, y in ds.take(1):
            assert x.shape[-1] == 1
            break

    def test_drop_remainder(self, npy_files):
        obs_files, true_files, _ = npy_files
        # 5 files, batch_size=3, drop_remainder=True -> 1 batch of 3
        ds = create_dataset(obs_files, true_files, batch_size=3,
                            field_size=FIELD_SIZE, norm_val=20.0,
                            drop_remainder=True, shuffle=False)
        batches = list(ds)
        assert len(batches) == 1
        assert batches[0][0].shape[0] == 3
