"""Data discovery and tf.data pipeline for the Medical MNIST dataset.

The Medical MNIST dataset is laid out as one folder per class containing JPEG
files. This module finds the dataset root, lists files and labels, splits them
into train/val/test partitions, and exposes a ``tf.data.Dataset`` builder that
yields normalised grayscale images sized for the autoencoder models.

Typical usage::

    from src.data_processing import find_data_root, list_paths_and_labels, \\
        split_paths, make_dataset

    data_root = find_data_root()
    classes, paths, labels = list_paths_and_labels(data_root)
    (tr_p, tr_y), (va_p, va_y), (te_p, te_y) = split_paths(paths, labels)
    train_ds = make_dataset(tr_p, tr_y, training=True)
"""

from __future__ import annotations

import glob
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
IMG_SIZE: int = 64
BATCH_SIZE: int = 128
SEED: int = 42

DEFAULT_DATA_ROOT_CANDIDATES: Tuple[str, ...] = (
    "/kaggle/input/medical-mnist",
    "/kaggle/input/datasets/andrewmvd/medical-mnist",
    "data/processed/medical-mnist",
    "data/raw/medical-mnist",
)


# ---------------------------------------------------------------------------
# Path discovery
# ---------------------------------------------------------------------------
def find_data_root(
    candidates: Optional[Sequence[str]] = None,
) -> str:
    """Locate the Medical MNIST root folder.

    The root is the directory that contains one sub-directory per class
    (``AbdomenCT``, ``BreastMRI``, ...). The function tries the provided
    candidate paths first, then scans ``/kaggle/input/*`` one and two levels
    deep.

    Args:
        candidates: Iterable of candidate paths to test, in priority order.
            If ``None``, ``DEFAULT_DATA_ROOT_CANDIDATES`` is used.

    Returns:
        The absolute path to the Medical MNIST root folder.

    Raises:
        FileNotFoundError: If no valid root could be located.
    """
    candidates = tuple(candidates) if candidates else DEFAULT_DATA_ROOT_CANDIDATES

    for candidate in candidates:
        if os.path.isdir(candidate) and any(
            os.path.isdir(os.path.join(candidate, child))
            for child in os.listdir(candidate)
        ):
            return candidate

    # Fallback: scan /kaggle/input one and two levels deep
    for root in glob.glob("/kaggle/input/*"):
        if os.path.isdir(os.path.join(root, "AbdomenCT")):
            return root
        for sub in glob.glob(os.path.join(root, "*")):
            if os.path.isdir(os.path.join(sub, "AbdomenCT")):
                return sub

    raise FileNotFoundError(
        "Medical MNIST dataset not found. Tried: "
        f"{list(candidates)} and /kaggle/input/*"
    )


# ---------------------------------------------------------------------------
# Listing and splitting
# ---------------------------------------------------------------------------
def list_paths_and_labels(
    data_root: str,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Return sorted class names, image-path array and integer-label array.

    Args:
        data_root: Path containing one sub-directory per class.

    Returns:
        Triple ``(class_names, paths, labels)`` where ``paths`` and ``labels``
        are 1-D numpy arrays of equal length.

    Raises:
        FileNotFoundError: If ``data_root`` does not exist.
        ValueError: If ``data_root`` contains no class sub-directories.
    """
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"data_root does not exist: {data_root}")

    class_names: List[str] = sorted(
        d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))
    )
    if not class_names:
        raise ValueError(f"No class sub-directories found in {data_root}")

    paths: List[str] = []
    labels: List[int] = []
    for class_index, class_name in enumerate(class_names):
        class_paths = sorted(glob.glob(os.path.join(data_root, class_name, "*")))
        paths.extend(class_paths)
        labels.extend([class_index] * len(class_paths))

    return class_names, np.array(paths), np.array(labels, dtype=np.int32)


def split_paths(
    paths: np.ndarray,
    labels: np.ndarray,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = SEED,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """Shuffle deterministically and split into train, val, test partitions.

    Args:
        paths: 1-D array of file paths.
        labels: 1-D array of integer labels parallel to ``paths``.
        train_frac: Fraction of the data to allocate to training.
        val_frac: Fraction to allocate to validation. The remainder goes to
            test. Must satisfy ``train_frac + val_frac < 1.0``.
        seed: PRNG seed for the deterministic shuffle.

    Returns:
        Three ``(paths, labels)`` pairs corresponding to train, val, and test.

    Raises:
        ValueError: If the fractions are inconsistent or the inputs do not have
            equal length.
    """
    if len(paths) != len(labels):
        raise ValueError("paths and labels must have equal length")
    if not 0.0 < train_frac < 1.0 or not 0.0 < val_frac < 1.0:
        raise ValueError("train_frac and val_frac must each be in (0, 1)")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.0")

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(paths))
    paths = paths[permutation]
    labels = labels[permutation]

    n_total = len(paths)
    n_train = int(train_frac * n_total)
    n_val = int(val_frac * n_total)

    train = (paths[:n_train], labels[:n_train])
    val = (paths[n_train : n_train + n_val], labels[n_train : n_train + n_val])
    test = (paths[n_train + n_val :], labels[n_train + n_val :])
    return train, val, test


# ---------------------------------------------------------------------------
# tf.data pipeline
# ---------------------------------------------------------------------------
def _decode_image(path: tf.Tensor) -> tf.Tensor:
    """Read a JPEG file from disk and return a normalised grayscale tensor.

    Args:
        path: Scalar string tensor pointing to the image file.

    Returns:
        Float32 tensor of shape ``(IMG_SIZE, IMG_SIZE, 1)`` with values in
        ``[0, 1]``.
    """
    raw = tf.io.read_file(path)
    image = tf.io.decode_jpeg(raw, channels=1)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    return tf.cast(image, tf.float32) / 255.0


def make_dataset(
    paths: np.ndarray,
    labels: np.ndarray,
    training: bool = False,
    with_labels: bool = False,
    batch_size: int = BATCH_SIZE,
    shuffle_buffer: int = 4096,
) -> tf.data.Dataset:
    """Build a batched, prefetched ``tf.data.Dataset`` from paths and labels.

    By default the dataset yields ``(image, image)`` pairs suitable for
    autoencoder training. Set ``with_labels=True`` to instead yield
    ``(image, label)`` pairs, which is useful for latent-space visualisation.

    Args:
        paths: 1-D array of file paths.
        labels: 1-D array of integer labels parallel to ``paths``.
        training: If True, the dataset is shuffled with a fixed seed.
        with_labels: If True, yield ``(image, label)`` instead of
            ``(image, image)``.
        batch_size: Mini-batch size.
        shuffle_buffer: Maximum shuffle buffer size when ``training`` is True.

    Returns:
        A configured ``tf.data.Dataset``, batched and prefetched.
    """
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        dataset = dataset.shuffle(
            buffer_size=min(len(paths), shuffle_buffer),
            seed=SEED,
        )

    if with_labels:
        dataset = dataset.map(
            lambda p, y: (_decode_image(p), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    else:
        # Autoencoder convention: target == input
        dataset = dataset.map(
            lambda p, y: (_decode_image(p), _decode_image(p)),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
