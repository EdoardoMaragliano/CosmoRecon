# 3D UNet for Cosmological Field Inpainting

This repository implements a 3D UNet-based model for **inpainting cosmological density fields**. The model reconstructs missing or corrupted regions in simulated density fields using observed inputs, optionally combined with masks. The project leverages TensorFlow, supports multi-GPU training with `MirroredStrategy`, and includes tools for hyperparameter optimization.

---

## Features

* 3D UNet architecture for volumetric data
* Optional use of masks for missing regions
* Data loading via `tf.data.Dataset` with shuffle, batch, repeat, and prefetch
* Multi-GPU training support
* Logging and checkpointing
* Hyperparameter optimization using Optuna
* Evaluation and prediction saving for validation datasets

---

## Requirements

* Python 3.9+
* TensorFlow 2.x
* NumPy
* scikit-learn
* Optuna (for hyperparameter tuning)
* Additional Python packages: `tqdm`, `PyYAML`, `colorlog`, `sqlalchemy`, `alembic`, `Mako`, `greenlet`

You can install dependencies via:

```bash
pip install -r requirements.txt
```

or manually:

```bash
pip install tensorflow numpy scikit-learn optuna tqdm PyYAML colorlog sqlalchemy alembic Mako greenlet
```

---

## Project Structure

```
├── InpaintingModel.py       # Model definition
├── train.py                 # Training script
├── EpochCheckpoint.py       # Custom callback for epoch-wise saving
├── output_products/         # Directory for saved models, logs, and predictions
├── obs/                     # Observed input fields (.npy)
├── true/                    # True target fields (.npy)
├── mask/                    # Optional mask files (.npy)
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Usage

### Training

```bash
python train.py \
    --obs_dir path/to/observed \
    --true_dir path/to/true \
    --mask_dir path/to/masks \
    --input_field rho \
    --output_dir output_products \
    --batch_size 16 \
    --epochs 500 \
    --repeat_dataset False \
    --drop_remainder False \
    --learning_rate 1e-4
```

### Options

* `--input_field`: `'rho'` (density) or `'delta'` (shifted ReLU output)
* `--use_mask`: Use mask for missing regions (`True` or `False`)
* `--repeat_dataset`: Repeat dataset indefinitely (useful for continuous training)
* `--drop_remainder`: Drop last incomplete batch for consistent batch sizes
* `--save_freq`: Save model every N epochs
* `--learning_rate`: Initial learning rate for Adam optimizer

---

### Multi-GPU Training

The model automatically uses all available GPUs with `MirroredStrategy`. You can adjust batch size depending on GPU memory:

```python
strategy = tf.distribute.MirroredStrategy()
```

---

### Hyperparameter Optimization

Optuna is integrated for tuning batch size and learning rate:

```bash
python optimize_hyperparameters.py
```

This will search for the **best combination** of batch size and learning rate based on validation MSE.

---

## Evaluation

After training, predicted fields are saved in `output_products/output_data/`:

```bash
python train.py --predict_only True
```

Predictions are normalized to match the original input scale.

---

## Logging

* Training and validation losses are saved in `output_products/losses/`
* TensorBoard logs are stored in `output_products/logs/`
* Models are saved per epoch in `output_products/store_models/`

---

## License

MIT License – see `LICENSE` for details.

---

## Contact

Edoardo Maragliano – [edoardo.maragliano@gmail.com](mailto:edoardo.maragliano@gmail.com)
