"""
CosmoRecon -- 3D cosmological field reconstruction with deep learning.

Public API
----------
Models:
    UNet, MaskedUNet3D

Loss functions:
    MaskedMSE, MaskedGradientLoss, MaskedMSEWithGradient

Data loading:
    create_dataset

Callbacks:
    SaveEveryNEpoch, EpochCheckpoint

Utilities:
    compute_gradient, compute_depth, dilate_mask, prepare_mask_tensor,
    shifted_relu, build_unet

Analysis & plotting (lazy-loaded, requires pypower + pmesh):
    OutputReader, Plotter, Plotter2D, set_latex_env

    To install analysis dependencies::

        pip install "CosmoRecon[analysis]"

    Or manually::

        pip install pypower pmesh
"""

# Core imports (always available)
from CosmoRecon.models import (
    UNet,
    MaskedUNet3D,
    MaskedInpaintingUNet,  # deprecated alias
    MaskedMSE,
    MaskedGradientLoss,
    MaskedMSEWithGradient,
    compute_gradient,
    compute_depth,
    dilate_mask,
    prepare_mask_tensor,
    shifted_relu,
    build_unet,
)
from CosmoRecon.datahandler import create_dataset
from CosmoRecon.checkpoints import SaveEveryNEpoch, EpochCheckpoint


def __getattr__(name: str):
    """Lazy-load analysis/plotting classes that depend on pypower and pmesh.

    This avoids forcing users who only need the training pipeline to install
    heavy analysis dependencies.
    """
    _analysis_names = {"OutputReader", "Plotter", "Plotter2D", "set_latex_env"}
    if name in _analysis_names:
        try:
            from CosmoRecon import OutputReaders as _mod
        except ImportError as e:
            raise ImportError(
                f"Cannot import '{name}' because analysis dependencies are "
                f"missing: {e}. Install them with: pip install pypower pmesh "
                f"(or: pip install 'CosmoRecon[analysis]')"
            ) from e
        return getattr(_mod, name)
    raise AttributeError(f"module 'CosmoRecon' has no attribute {name!r}")


__all__ = [
    # Models
    "UNet",
    "MaskedUNet3D",
    "MaskedInpaintingUNet",  # deprecated alias
    # Losses
    "MaskedMSE",
    "MaskedGradientLoss",
    "MaskedMSEWithGradient",
    # Data
    "create_dataset",
    # Callbacks
    "SaveEveryNEpoch",
    "EpochCheckpoint",
    # Analysis (lazy)
    "OutputReader",
    "Plotter",
    "Plotter2D",
    "set_latex_env",
    # Utilities
    "compute_gradient",
    "compute_depth",
    "dilate_mask",
    "prepare_mask_tensor",
    "shifted_relu",
    "build_unet",
]
