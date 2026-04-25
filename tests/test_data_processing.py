"""Unit tests for ``src.data_processing``.

These tests build a tiny synthetic Medical-MNIST-like dataset in a temporary
directory, then exercise the discovery, listing, splitting, and tf.data
pipeline functions. They do not require the real dataset and run in a few
seconds on CPU.

Run from the repository root::

    pytest tests/test_data_processing.py -v
"""

from __future__ import annotations

import os
import tempfile
from typing import Iterator

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from src.data_processing import (
    BATCH_SIZE,
    IMG_SIZE,
    find_data_root,
    list_paths_and_labels,
    make_dataset,
    split_paths,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
SYNTHETIC_CLASSES = ("AbdomenCT", "BreastMRI", "CXR", "ChestCT", "Hand", "HeadCT")
IMAGES_PER_CLASS = 20


@pytest.fixture
def synthetic_dataset_root() -> Iterator[str]:
    """Create a tiny synthetic Medical-MNIST-like dataset in a temp dir.

    Yields the path to the dataset root. The directory is removed
    automatically when the test completes.
    """
    with tempfile.TemporaryDirectory() as tmp_root:
        for class_name in SYNTHETIC_CLASSES:
            class_dir = os.path.join(tmp_root, class_name)
            os.makedirs(class_dir, exist_ok=True)
            for image_index in range(IMAGES_PER_CLASS):
                # Random grayscale 64x64 image saved as JPEG
                pixels = np.random.randint(
                    0, 255, size=(IMG_SIZE, IMG_SIZE), dtype=np.uint8
                )
                Image.fromarray(pixels, mode="L").save(
                    os.path.join(class_dir, f"sample_{image_index:03d}.jpeg")
                )
        yield tmp_root


# ---------------------------------------------------------------------------
# find_data_root
# ---------------------------------------------------------------------------
def test_find_data_root_uses_provided_candidate(synthetic_dataset_root: str) -> None:
    """``find_data_root`` returns the first valid candidate."""
    found = find_data_root(candidates=[synthetic_dataset_root])
    assert found == synthetic_dataset_root


def test_find_data_root_raises_when_no_root(tmp_path) -> None:
    """``find_data_root`` raises ``FileNotFoundError`` when nothing exists."""
    with pytest.raises(FileNotFoundError):
        # Use a path that definitely does not contain a Medical MNIST tree
        find_data_root(candidates=[str(tmp_path / "does_not_exist")])


# ---------------------------------------------------------------------------
# list_paths_and_labels
# ---------------------------------------------------------------------------
def test_list_paths_and_labels_returns_correct_counts(
    synthetic_dataset_root: str,
) -> None:
    """All synthetic images are listed and labels are class-aligned."""
    classes, paths, labels = list_paths_and_labels(synthetic_dataset_root)
    assert classes == sorted(SYNTHETIC_CLASSES)
    assert len(paths) == len(SYNTHETIC_CLASSES) * IMAGES_PER_CLASS
    assert len(paths) == len(labels)
    for class_index in range(len(classes)):
        assert (labels == class_index).sum() == IMAGES_PER_CLASS


def test_list_paths_and_labels_raises_on_missing_root() -> None:
    """An invalid root raises ``FileNotFoundError``."""
    with pytest.raises(FileNotFoundError):
        list_paths_and_labels("/path/that/does/not/exist")


def test_list_paths_and_labels_raises_on_empty_root(tmp_path) -> None:
    """A valid but empty root raises ``ValueError``."""
    with pytest.raises(ValueError):
        list_paths_and_labels(str(tmp_path))


# ---------------------------------------------------------------------------
# split_paths
# ---------------------------------------------------------------------------
def test_split_paths_partition_sizes_are_correct(
    synthetic_dataset_root: str,
) -> None:
    """Train/val/test sizes match the requested fractions and cover the data."""
    _, paths, labels = list_paths_and_labels(synthetic_dataset_root)
    total = len(paths)
    (tr_p, tr_y), (va_p, va_y), (te_p, te_y) = split_paths(
        paths, labels, train_frac=0.8, val_frac=0.1
    )
    assert len(tr_p) == int(0.8 * total)
    assert len(va_p) == int(0.1 * total)
    assert len(tr_p) + len(va_p) + len(te_p) == total
    # No overlap between partitions
    assert set(tr_p).isdisjoint(va_p)
    assert set(tr_p).isdisjoint(te_p)
    assert set(va_p).isdisjoint(te_p)


def test_split_paths_is_deterministic(synthetic_dataset_root: str) -> None:
    """Two splits with the same seed produce identical partitions."""
    _, paths, labels = list_paths_and_labels(synthetic_dataset_root)
    first = split_paths(paths, labels, seed=123)
    second = split_paths(paths, labels, seed=123)
    np.testing.assert_array_equal(first[0][0], second[0][0])
    np.testing.assert_array_equal(first[1][0], second[1][0])
    np.testing.assert_array_equal(first[2][0], second[2][0])


def test_split_paths_validates_fractions(synthetic_dataset_root: str) -> None:
    """Bad fraction combinations raise ``ValueError``."""
    _, paths, labels = list_paths_and_labels(synthetic_dataset_root)
    with pytest.raises(ValueError):
        split_paths(paths, labels, train_frac=0.9, val_frac=0.2)


def test_split_paths_validates_lengths() -> None:
    """Mismatched ``paths`` and ``labels`` lengths raise ``ValueError``."""
    with pytest.raises(ValueError):
        split_paths(np.array(["a", "b"]), np.array([0]))


# ---------------------------------------------------------------------------
# make_dataset
# ---------------------------------------------------------------------------
def test_make_dataset_yields_correct_image_shape(
    synthetic_dataset_root: str,
) -> None:
    """Pipeline yields ``(image, image)`` tensors with the expected shape."""
    _, paths, labels = list_paths_and_labels(synthetic_dataset_root)
    dataset = make_dataset(paths, labels, batch_size=4)
    x, y = next(iter(dataset))
    assert x.shape == (4, IMG_SIZE, IMG_SIZE, 1)
    assert y.shape == (4, IMG_SIZE, IMG_SIZE, 1)
    # Reconstruction targets equal inputs for autoencoder training
    np.testing.assert_array_equal(x.numpy(), y.numpy())


def test_make_dataset_with_labels(synthetic_dataset_root: str) -> None:
    """``with_labels=True`` yields ``(image, integer_label)`` pairs."""
    _, paths, labels = list_paths_and_labels(synthetic_dataset_root)
    dataset = make_dataset(paths, labels, with_labels=True, batch_size=4)
    x, y = next(iter(dataset))
    assert x.shape == (4, IMG_SIZE, IMG_SIZE, 1)
    assert y.shape == (4,)
    assert y.dtype == tf.int32


def test_make_dataset_normalises_to_unit_range(
    synthetic_dataset_root: str,
) -> None:
    """Image tensors are float32 normalised into ``[0, 1]``."""
    _, paths, labels = list_paths_and_labels(synthetic_dataset_root)
    dataset = make_dataset(paths, labels, batch_size=4)
    x, _ = next(iter(dataset))
    assert x.dtype == tf.float32
    assert tf.reduce_min(x).numpy() >= 0.0
    assert tf.reduce_max(x).numpy() <= 1.0
