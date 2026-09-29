"""Helpers for reproducible dataset splits, training, and inference."""

import os
import pickle
import random
import warnings
from pathlib import Path

import numpy as np


def split_subjects(subjects, seed=2026):
    """Create the existing 64/16/20 patient split reproducibly."""
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")

    subjects = np.unique(np.asarray(subjects))
    shuffled_subjects = np.random.default_rng(seed).permutation(subjects)
    n_subjects = len(shuffled_subjects)
    train_end = int(0.64 * n_subjects)
    valid_end = int(0.80 * n_subjects)
    return (
        shuffled_subjects[:train_end],
        shuffled_subjects[train_end:valid_end],
        shuffled_subjects[valid_end:],
    )


def split_validation_subjects(subjects, seed=2026):
    """Split the legacy validation patients evenly into model-selection and calibration sets."""
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    subjects = np.unique(np.asarray(subjects))
    ordered = np.random.default_rng(seed + 1).permutation(subjects)
    midpoint = (len(ordered) + 1) // 2
    return ordered[:midpoint], ordered[midpoint:]


def seed_everything(seed=2026, deterministic=True):
    """Seed Python, NumPy and PyTorch, and request deterministic kernels.

    Exact reproducibility is expected for the same inputs, software versions,
    device and hardware; it is not guaranteed across PyTorch/CUDA versions or
    different hardware.
    """
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")

    # cuBLAS reads this when its CUDA context is initialized. Set it before
    # making any CUDA calls (including manual_seed_all below).
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = not deterministic
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = not deterministic
    torch.use_deterministic_algorithms(deterministic)


def validate_split_seed(split_path, expected_seed):
    """Warn for legacy splits or reject a split generated with another seed."""
    split_path = Path(split_path)
    if not split_path.exists():
        warnings.warn(
            "Split metadata not found at '{}'; its seed cannot be verified. "
            "Regenerate the dataset with preprocess/step_4.py for reproducible splits.".format(split_path),
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    with split_path.open("rb") as file:
        metadata = pickle.load(file)

    split_seed = metadata.get("seed") if isinstance(metadata, dict) else None
    if split_seed is None:
        warnings.warn(
            "Split metadata at '{}' has no seed; its reproducibility cannot be verified.".format(split_path),
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    split_seed = int(split_seed)
    if split_seed != int(expected_seed):
        raise ValueError(
            "The prepared patient split uses seed {}, but this run uses seed {}. "
            "Regenerate it with `python preprocess/step_4.py --seed {}` or pass "
            "the matching --seed.".format(split_seed, expected_seed, expected_seed)
        )
    return split_seed
