"""Tests for the GPU configuration utility (D3)."""

import os
import sys
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from CosmoRecon.utils.gpu import configure_gpus


class TestConfigureGpus:
    """Tests for ``configure_gpus``."""

    def test_no_gpus_returns_empty(self):
        with patch("CosmoRecon.utils.gpu.tf.config") as mock_config:
            mock_config.list_physical_devices.return_value = []
            result = configure_gpus()
            assert result == []

    def test_no_gpus_logs_warning(self, caplog):
        import logging
        with patch("CosmoRecon.utils.gpu.tf.config") as mock_config:
            mock_config.list_physical_devices.return_value = []
            with caplog.at_level(logging.WARNING):
                configure_gpus()
            assert any("No GPUs" in r.message for r in caplog.records)

    def test_out_of_range_index_warns(self, caplog):
        import logging
        fake_gpu = MagicMock()
        with patch("CosmoRecon.utils.gpu.tf.config") as mock_config:
            mock_config.list_physical_devices.return_value = [fake_gpu]
            mock_config.set_visible_devices = MagicMock()
            mock_config.experimental.set_memory_growth = MagicMock()
            with caplog.at_level(logging.WARNING):
                configure_gpus(device_indices=[5])
            assert any("out of range" in r.message for r in caplog.records)
