#!/usr/bin/env python3
"""Remove generated artefacts from a training run directory.

Usage::

    python clean_outputs.py [--dir <training_run_dir>]

Defaults to ``train_delta`` if no directory is specified.
"""

import argparse
import os
from glob import glob


def clean(run_dir):
    """Delete log, model, data, loss and cache files under *run_dir*."""
    subdirs = ['logs/*/', 'store_models', 'output_data', 'losses', '__pycache__']
    removed = 0
    for pattern in subdirs:
        for f in glob(os.path.join(run_dir, pattern, '*')):
            if os.path.isfile(f):
                os.remove(f)
                print(f"Removed: {f}")
                removed += 1

    logfile = os.path.join(run_dir, 'train.log')
    if os.path.isfile(logfile):
        os.remove(logfile)
        print(f"Removed: {logfile}")
        removed += 1

    print(f"Cleanup completed ({removed} files removed)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean training run artefacts")
    parser.add_argument(
        '--dir', type=str, default='train_delta',
        help='Training run directory to clean (default: train_delta)',
    )
    args = parser.parse_args()
    clean(args.dir)
