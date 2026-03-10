# CosmoRecon

**Deep-learning reconstruction of 3D cosmological density fields**

CosmoRecon learns mappings between 3D density fields -- such as removing redshift-space distortions, reconstructing masked regions, or denoising -- using an adaptive 3D U-Net with optional mask-aware losses.

---

## Highlights

- **Adaptive 3D U-Net** -- network depth is determined automatically from the input grid size, so the same architecture works for 64^3 through 256^3 volumes (and beyond).
- **Mask-aware training** -- custom `MaskedMSE` and `MaskedGradientLoss` losses focus the optimisation on selected voxels, leaving observed regions untouched.
- **Multi-GPU support** -- training scripts leverage `tf.distribute.MirroredStrategy` out of the box.
- **Flexible I/O** -- data is streamed from `.npy` files through `tf.data` pipelines with shuffling, batching, and prefetching, keeping memory usage constant regardless of dataset size.
- **Hyper-parameter search** -- built-in Optuna integration for tuning batch size, learning rate, and other training hyper-parameters.
- **Publication-quality analysis** -- `OutputReader` computes power-spectrum multipoles (P_0, P_2, P_4), residuals, and chi-squared statistics; `Plotter` renders them with LaTeX labels ready for papers.

---

## Repository Structure

```
3DFieldReconstruction/
|
|-- CosmoRecon/                     # Python package
|   |-- CosmoRecon/                 # Core library
|   |   |-- __init__.py             # Public API exports
|   |   |-- models.py              # UNet, MaskedUNet3D, loss functions
|   |   |-- datahandler.py          # tf.data dataset creation
|   |   |-- checkpoints.py          # Periodic model-saving callbacks
|   |   |-- OutputReaders.py        # Post-hoc analysis (Pk multipoles, plotting)
|   |   +-- utils/
|   |       |-- __init__.py
|   |       |-- gpu.py              # GPU configuration helper
|   |       +-- loggers.py          # Logging setup helper
|   |
|   |-- scripts/                    # Entry-point training & evaluation scripts
|   |   |-- train.py                # Unified training (MSE / masked MSE / masked gradient)
|   |   |-- restart_training_from_saved_model.py
|   |   |-- evaluate_model.py
|   |   |-- optimize_hyperpars.py
|   |   +-- clean_outputs.py
|   |
|   |-- tests/                      # pytest test suite
|   |-- examples/                   # Jupyter notebook examples
|   |-- params.json                 # Example parameter file
|   +-- pipeline_diagram.md         # ASCII pipeline overview
|
|-- training_runs/                  # (gitignored) saved checkpoints & logs
|-- optuna_runs/                    # (gitignored) Optuna study outputs
|-- tf_environment.yml              # Conda environment specification
|-- readme.md                       # This file
+-- .gitignore
```

---

## Quick Start

### 1. Environment Setup

```bash
conda env create -f tf_environment.yml
conda activate tf
```

Or install the core dependencies manually:

```bash
pip install tensorflow numpy scikit-learn optuna tqdm matplotlib
```

### 2. Prepare Your Data

Organise your density fields as individual `.npy` files:

```
data/
  observed/      # Input fields (e.g. redshift-space density)
    mock_000.npy
    mock_001.npy
    ...
  true/           # Target fields (e.g. real-space density)
    mock_000.npy
    mock_001.npy
    ...
  masks/          # (Optional) binary mask(s): 1 = observed, 0 = missing
    mask.npy
```

### 3. Configure Parameters

Create a JSON file (or use `CosmoRecon/params.json` as a template):

```json
{
  "obs_dir": "data/observed/",
  "true_dir": "data/true/",
  "mask_dir": "data/masks/",
  "output_dir": "runs/my_experiment/",
  "field_size": 128,
  "input_field": "rho",
  "batch_size": 16,
  "epochs": 500,
  "learning_rate": 1e-4,
  "use_mask": true,
  "density_normalization": 40.0
}
```

### 4. Train

**Standard MSE training** (no mask):

```bash
python CosmoRecon/scripts/train.py \
    --param_file params.json --loss_type mse
```

**Masked MSE training** (loss only on selected voxels):

```bash
python CosmoRecon/scripts/train.py \
    --param_file params.json --loss_type masked_mse --use_mask
```

**Masked MSE + gradient loss** (smoother boundaries):

```bash
python CosmoRecon/scripts/train.py \
    --param_file params.json --loss_type masked_gradient --use_mask
```

**Resume from checkpoint:**

```bash
python CosmoRecon/scripts/restart_training_from_saved_model.py \
    --param_file params.json \
    --resume_from runs/my_experiment/store_models/model_100.keras
```

#### GPU Configuration

**By default, training distributes across ALL available GPUs** using `tf.distribute.MirroredStrategy`.

To limit which GPUs are used, specify them with `--gpu_indices`:

```bash
# Use only GPUs 2 and 3
python CosmoRecon/scripts/train.py \
    --param_file params.json \
    --gpu_indices 2,3
```

Alternatively, use the `CUDA_VISIBLE_DEVICES` environment variable:

```bash
# Use only GPUs 2 and 3
CUDA_VISIBLE_DEVICES=2,3 python CosmoRecon/scripts/train.py \
    --param_file params.json
```

**Note:** When using `CUDA_VISIBLE_DEVICES`, TensorFlow will renumber the visible devices starting from 0 (e.g., physical GPUs 2,3 become logical GPUs 0,1 in the logs).

### 5. Evaluate

```bash
python CosmoRecon/scripts/evaluate_model.py \
    --param_file params.json \
    --model_path runs/my_experiment/store_models/model_500.keras \
    --output_dir runs/my_experiment/eval
```

### 6. Hyper-parameter Optimisation

Edit the configuration section in `optimize_hyperpars.py`, then:

```bash
python CosmoRecon/scripts/optimize_hyperpars.py
```

---

## Architecture

```
Input (N, N, N, C)
       |
  [ Encoder ]  Conv3D -> Conv3D -> MaxPool3D  (x depth)
       |
  [ Bottleneck ]  Conv3D -> Conv3D -> (Dropout)
       |
  [ Decoder ]  Conv3D -> Conv3D -> Conv3DTranspose -> Concat(skip)  (x depth)
       |
  Conv3D(1, activation=ReLU)
       |
Output (N, N, N, 1)
```

- **Depth** is computed as `floor(log2(N / min_size))`.
- Filters start at `base_filters` and double at each encoder level.
- For delta-field reconstruction, a **shifted ReLU** enforces the physical constraint delta >= -1.

---

## Loss Functions

| Loss | Description |
|------|-------------|
| `MaskedMSE` | MSE restricted to masked voxels (mask == 0) |
| `MaskedGradientLoss` | Finite-difference gradient matching near mask boundaries |
| `MaskedMSEWithGradient` | Weighted combination of the above |

---

## Analysis Pipeline

```python
from CosmoRecon import OutputReader, Plotter, set_latex_env

set_latex_env()

reader = OutputReader(
    real_space="data/real/*.npy",
    redshift_space="data/rsd/*.npy",
    nn_rec="runs/eval/output_data/*.npy",
)
reader.load_fields()
reader.compute_all_stats(
    modes=["redshift_space", "nn_rec"],
    grid_size=128, box_size=1000.0, box_centre=[500, 500, 500],
)

plotter = Plotter(reader)
fig = plotter.plot_pk_multipoles_and_residuals()
fig.savefig("pk_comparison.pdf", bbox_inches="tight")
```

---

## Key CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--param_file` | -- | Path to JSON config (overrides CLI args) |
| `--loss_type` | `mse` | `mse`, `masked_mse`, or `masked_gradient` |
| `--input_field` | `rho` | `rho` (density) or `delta` (contrast, shifted ReLU) |
| `--use_mask` | `False` | Enable mask-aware training |
| `--field_size` | `128` | Spatial grid size N |
| `--batch_size` | `16` | Training batch size |
| `--epochs` | `500` | Maximum training epochs |
| `--learning_rate` | `1e-4` | Adam learning rate |
| `--density_normalization` | `40.0` | Divide fields by this value before training |
| `--gpu_indices` | all available | Comma-separated GPU indices (e.g. `2,3` to use only GPUs 2 and 3) |
| `--debug` | `False` | Enable verbose debug logging |
| `--resume_from` | -- | `.keras` checkpoint to resume training from |

---

## Testing

The project includes a pytest test suite under `CosmoRecon/tests/` covering training helpers, data handling, checkpointing, evaluation, and training restart workflows. All tests run on CPU using small synthetic 16^3 fields.

### Running the tests

```bash
cd CosmoRecon
python -m pytest tests/ -q
```

To skip GPU-only tests (if any are marked):

```bash
python -m pytest tests/ -q -m "not gpu"
```

### Test structure

| File | What it covers |
|------|----------------|
| `conftest.py` | Shared fixtures: synthetic data generation, script importers, CLI arg builders |
| `test_train_helpers.py` | `_parse_gpu_indices`, `_load_json_params`, `_set_random_seeds`, `build_parser` |
| `test_train_validation.py` | Input validation: missing dirs, file mismatches, mask requirements, index slicing |
| `test_train_integration.py` | End-to-end training: MSE, masked MSE, masked gradient, seed reproducibility, delta mode, dropout |
| `test_datahandler.py` | `create_dataset` shapes, normalization, channel modes, mask handling |
| `test_checkpoints.py` | `SaveEveryNEpoch` intervals, best-only saving, mode validation, `EpochCheckpoint` deprecation |
| `test_gpu_config.py` | `configure_gpus` with mocked devices: empty GPU list, out-of-range indices |
| `test_evaluate.py` | Evaluation validation errors, MSE/masked end-to-end eval, split consistency, JSON warnings |
| `test_restart.py` | Epoch extraction, mask re-injection roundtrip, CSV append, resume/fresh-start end-to-end |

### Dependencies

Tests require `pytest` (listed in the `[dev]` optional dependency group):

```bash
pip install -e ".[dev]"
```

---

## Outputs

After training, the output directory contains:

```
<output_dir>/
  store_models/       # Periodic checkpoints (model_*.keras)
  logs/               # Training logs + TensorBoard events
  losses/             # loss.npy, val_loss.npy
  output_data/        # Predicted fields (pred_field_*.npy)
  history.csv         # Epoch-by-epoch metrics
```

---

## Requirements

- Python 3.9+
- TensorFlow 2.18+
- NumPy, scikit-learn, matplotlib, tqdm
- Optuna (for hyper-parameter search)
- pypower, pmesh (for power-spectrum analysis)

A full Conda environment spec is in `tf_environment.yml`.

---

## License

MIT License -- see `LICENSE` for details.

---

## Author

**Edoardo Maragliano** -- [edoardo.maragliano@gmail.com](mailto:edoardo.maragliano@gmail.com)
