"""GPU configuration utilities for CosmoRecon."""

import logging
from typing import Optional, List

import tensorflow as tf

logger = logging.getLogger(__name__)


def configure_gpus(
    device_indices: Optional[List[int]] = None,
    memory_growth: bool = True,
) -> List[tf.config.PhysicalDevice]:
    """Configure visible GPU devices and memory growth settings.

    Parameters
    ----------
    device_indices : list of int or None
        Indices of physical GPUs to make visible.  ``None`` means use all
        available GPUs.
    memory_growth : bool
        Whether to enable memory growth (allocate GPU memory on demand
        rather than pre-allocating the full GPU memory).

    Returns
    -------
    list of tf.config.PhysicalDevice
        The visible GPU devices after configuration.
    """
    physical_gpus = tf.config.list_physical_devices('GPU')
    if not physical_gpus:
        logger.warning("No GPUs detected.")
        return []

    try:
        if device_indices is not None:
            selected = []
            for idx in device_indices:
                if idx < len(physical_gpus):
                    selected.append(physical_gpus[idx])
                else:
                    logger.warning(
                        "GPU index %d out of range (only %d GPUs available), skipping.",
                        idx, len(physical_gpus),
                    )
            if selected:
                tf.config.set_visible_devices(selected, 'GPU')
            else:
                logger.warning("No valid GPU indices; using all GPUs.")
        else:
            tf.config.set_visible_devices(physical_gpus, 'GPU')

        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, memory_growth)

        visible = tf.config.list_physical_devices('GPU')
        logger.info(
            "Configured %d GPU(s), memory_growth=%s", len(visible), memory_growth,
        )
        return visible

    except RuntimeError as e:
        logger.error("GPU configuration failed: %s", e)
        return tf.config.list_physical_devices('GPU')
