"""End-to-end training entry point for the AE and beta-VAE.

Run from the repository root::

    python -m src.train --data-root data/raw/medical-mnist --epochs 30

Or with all defaults (Kaggle layout)::

    python -m src.train

The script:
    1. Locates the dataset.
    2. Builds train / val / test ``tf.data`` pipelines.
    3. Trains the AE and the beta-VAE for a configurable number of epochs.
    4. Saves trained weights under ``models/`` and a metrics file under
       ``figures/metrics.txt``.

It deliberately does NOT regenerate the figures -- those are produced by the
notebook so the figure-generation code can stay close to the visual analysis.
The notebook and this module share the same modules in ``src/`` so they
cannot drift apart.
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import keras
import numpy as np
import tensorflow as tf

from src.data_processing import (
    BATCH_SIZE,
    SEED,
    find_data_root,
    list_paths_and_labels,
    make_dataset,
    split_paths,
)
from src.model import VAE, build_autoencoder, build_vae_decoder, build_vae_encoder

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_EPOCHS: int = 30
DEFAULT_KL_WEIGHT: float = 0.25
DEFAULT_LEARNING_RATE: float = 1e-3


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train AE and beta-VAE on Medical MNIST."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Path to the Medical MNIST root folder. If omitted, "
             "auto-detects on Kaggle / local conventions.",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory where trained model weights are saved.",
    )
    parser.add_argument(
        "--metrics-path",
        type=str,
        default="figures/metrics.txt",
        help="File where test-set MSE / MAE are written.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs (applies to both models).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--kl-weight",
        type=float,
        default=DEFAULT_KL_WEIGHT,
        help="beta coefficient for the VAE KL term. Default 0.25 "
             "(beta-VAE favouring reconstruction).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_global_seed(seed: int) -> None:
    """Seed numpy and tensorflow PRNGs for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def compute_recon_metrics(
    model: keras.Model, dataset: tf.data.Dataset
) -> Tuple[float, float]:
    """Compute mean MSE and MAE over a reconstruction dataset.

    Args:
        model: A model whose ``call`` returns reconstructions of its inputs.
        dataset: A ``tf.data.Dataset`` yielding ``(x, x)`` pairs.

    Returns:
        Pair ``(mse, mae)`` of pixelwise mean errors over the full dataset.
    """
    mses = []
    maes = []
    for x, _ in dataset:
        x_hat = model(x, training=False)
        mses.append(tf.reduce_mean(tf.square(x - x_hat)).numpy())
        maes.append(tf.reduce_mean(tf.abs(x - x_hat)).numpy())
    return float(np.mean(mses)), float(np.mean(maes))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the full AE + beta-VAE training pipeline."""
    args = parse_args()
    set_global_seed(args.seed)

    # ------ Data ----------------------------------------------------------
    data_root = args.data_root or find_data_root()
    print(f"[train] data_root = {data_root}")

    class_names, paths, labels = list_paths_and_labels(data_root)
    print(f"[train] classes = {class_names}")
    print(f"[train] total images = {len(paths)}")

    (train_p, train_y), (val_p, val_y), (test_p, test_y) = split_paths(
        paths, labels, seed=args.seed
    )
    print(
        f"[train] split  ->  train={len(train_p)}  "
        f"val={len(val_p)}  test={len(test_p)}"
    )

    train_ds = make_dataset(
        train_p, train_y, training=True, batch_size=args.batch_size
    )
    val_ds = make_dataset(val_p, val_y, batch_size=args.batch_size)
    test_ds = make_dataset(test_p, test_y, batch_size=args.batch_size)

    # ------ Output directories -------------------------------------------
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_path) or ".", exist_ok=True)

    # ------ Train AE -----------------------------------------------------
    print("\n[train] === Training AE ===")
    autoencoder, _ = build_autoencoder()
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(args.learning_rate),
        loss="binary_crossentropy",
    )
    autoencoder.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs, verbose=2
    )
    autoencoder.save(os.path.join(args.models_dir, "ae.keras"))

    # ------ Train VAE ----------------------------------------------------
    print(f"\n[train] === Training beta-VAE (beta={args.kl_weight}) ===")
    vae_encoder = build_vae_encoder()
    vae_decoder = build_vae_decoder()
    vae = VAE(vae_encoder, vae_decoder, kl_weight=args.kl_weight)
    vae.compile(optimizer=keras.optimizers.Adam(args.learning_rate))
    vae.fit(train_ds, validation_data=val_ds, epochs=args.epochs, verbose=2)
    vae_encoder.save(os.path.join(args.models_dir, "vae_encoder.keras"))
    vae_decoder.save(os.path.join(args.models_dir, "vae_decoder.keras"))

    # ------ Evaluate -----------------------------------------------------
    ae_mse, ae_mae = compute_recon_metrics(autoencoder, test_ds)
    vae_mse, vae_mae = compute_recon_metrics(vae, test_ds)

    print("\n[train] === Test metrics ===")
    print(f"AE  test MSE: {ae_mse:.6f}  MAE: {ae_mae:.6f}")
    print(f"VAE test MSE: {vae_mse:.6f}  MAE: {vae_mae:.6f}")

    with open(args.metrics_path, "w", encoding="utf-8") as f:
        f.write(f"AE  test MSE: {ae_mse:.6f}\n")
        f.write(f"AE  test MAE: {ae_mae:.6f}\n")
        f.write(f"VAE test MSE: {vae_mse:.6f}\n")
        f.write(f"VAE test MAE: {vae_mae:.6f}\n")
    print(f"[train] metrics written to {args.metrics_path}")


if __name__ == "__main__":
    main()
