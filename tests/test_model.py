"""Unit tests for ``src.model``.

These tests build the AE and beta-VAE on CPU, run a single forward and
backward pass on a tiny synthetic batch, and check shapes and finiteness.
They do not load the real dataset and complete in a few seconds on CPU.

Run from the repository root::

    pytest tests/test_model.py -v
"""

from __future__ import annotations

import keras
import numpy as np
import pytest
import tensorflow as tf

from src.data_processing import IMG_SIZE
from src.model import (
    AE_BOTTLENECK_CHANNELS,
    LATENT_DIM,
    VAE,
    Sampling,
    build_autoencoder,
    build_vae_decoder,
    build_vae_encoder,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
BATCH_SIZE_FOR_TEST = 4


@pytest.fixture
def synthetic_batch() -> tf.Tensor:
    """A tiny float32 batch with values in ``[0, 1]``."""
    rng = np.random.default_rng(0)
    array = rng.random(
        size=(BATCH_SIZE_FOR_TEST, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32
    )
    return tf.constant(array)


# ---------------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------------
def test_build_autoencoder_returns_two_models() -> None:
    """``build_autoencoder`` returns a paired (full, encoder-only) model."""
    autoencoder, encoder = build_autoencoder()
    assert isinstance(autoencoder, keras.Model)
    assert isinstance(encoder, keras.Model)


def test_autoencoder_output_shape_matches_input(
    synthetic_batch: tf.Tensor,
) -> None:
    """The decoder reconstructs the input resolution and channel count."""
    autoencoder, _ = build_autoencoder()
    output = autoencoder(synthetic_batch, training=False)
    assert output.shape == synthetic_batch.shape


def test_ae_encoder_bottleneck_shape(synthetic_batch: tf.Tensor) -> None:
    """The encoder produces an ``(8, 8, AE_BOTTLENECK_CHANNELS)`` tensor."""
    _, encoder = build_autoencoder()
    encoded = encoder(synthetic_batch, training=False)
    assert encoded.shape == (
        BATCH_SIZE_FOR_TEST,
        8,
        8,
        AE_BOTTLENECK_CHANNELS,
    )


def test_autoencoder_output_in_unit_range(synthetic_batch: tf.Tensor) -> None:
    """Sigmoid output is bounded to ``[0, 1]``."""
    autoencoder, _ = build_autoencoder()
    output = autoencoder(synthetic_batch, training=False).numpy()
    assert output.min() >= 0.0
    assert output.max() <= 1.0


def test_autoencoder_compiles_and_trains_one_step(
    synthetic_batch: tf.Tensor,
) -> None:
    """The AE compiles with BCE and runs one training step without error."""
    autoencoder, _ = build_autoencoder()
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
    history = autoencoder.fit(
        synthetic_batch,
        synthetic_batch,
        epochs=1,
        batch_size=BATCH_SIZE_FOR_TEST,
        verbose=0,
    )
    loss = history.history["loss"][0]
    assert np.isfinite(loss)


# ---------------------------------------------------------------------------
# VAE -- components
# ---------------------------------------------------------------------------
def test_sampling_layer_output_shape() -> None:
    """``Sampling`` returns a tensor of the same shape as its inputs."""
    z_mean = tf.zeros((BATCH_SIZE_FOR_TEST, LATENT_DIM))
    z_log_var = tf.zeros((BATCH_SIZE_FOR_TEST, LATENT_DIM))
    z = Sampling()([z_mean, z_log_var])
    assert z.shape == (BATCH_SIZE_FOR_TEST, LATENT_DIM)


def test_sampling_layer_is_stochastic() -> None:
    """Two calls with identical inputs return different samples."""
    z_mean = tf.zeros((BATCH_SIZE_FOR_TEST, LATENT_DIM))
    z_log_var = tf.zeros((BATCH_SIZE_FOR_TEST, LATENT_DIM))
    sampler = Sampling()
    z1 = sampler([z_mean, z_log_var]).numpy()
    z2 = sampler([z_mean, z_log_var]).numpy()
    assert not np.allclose(z1, z2)


def test_vae_encoder_outputs_three_tensors(synthetic_batch: tf.Tensor) -> None:
    """The encoder returns ``(z_mean, z_log_var, z)`` with the expected shape."""
    encoder = build_vae_encoder()
    z_mean, z_log_var, z = encoder(synthetic_batch, training=False)
    expected_shape = (BATCH_SIZE_FOR_TEST, LATENT_DIM)
    assert z_mean.shape == expected_shape
    assert z_log_var.shape == expected_shape
    assert z.shape == expected_shape


def test_vae_decoder_output_shape() -> None:
    """The decoder maps a latent batch back to the input image resolution."""
    decoder = build_vae_decoder()
    z = tf.random.normal((BATCH_SIZE_FOR_TEST, LATENT_DIM))
    out = decoder(z, training=False)
    assert out.shape == (BATCH_SIZE_FOR_TEST, IMG_SIZE, IMG_SIZE, 1)


def test_vae_decoder_output_in_unit_range() -> None:
    """Decoder sigmoid outputs are bounded to ``[0, 1]``."""
    decoder = build_vae_decoder()
    z = tf.random.normal((BATCH_SIZE_FOR_TEST, LATENT_DIM))
    out = decoder(z, training=False).numpy()
    assert out.min() >= 0.0
    assert out.max() <= 1.0


# ---------------------------------------------------------------------------
# VAE -- full model
# ---------------------------------------------------------------------------
def test_vae_call_reconstructs_input_shape(synthetic_batch: tf.Tensor) -> None:
    """``VAE.call`` returns reconstructions of the same shape as inputs."""
    vae = VAE(build_vae_encoder(), build_vae_decoder(), kl_weight=0.25)
    reconstruction = vae(synthetic_batch, training=False)
    assert reconstruction.shape == synthetic_batch.shape


def test_vae_compute_losses_returns_finite_scalars(
    synthetic_batch: tf.Tensor,
) -> None:
    """The three loss components are finite scalar tensors."""
    vae = VAE(build_vae_encoder(), build_vae_decoder(), kl_weight=0.25)
    z_mean, z_log_var, z = vae.encoder(synthetic_batch, training=False)
    x_hat = vae.decoder(z, training=False)
    total, rec, kl = vae._compute_losses(  # noqa: SLF001 (test internal API)
        synthetic_batch, x_hat, z_mean, z_log_var
    )
    for value in (total, rec, kl):
        assert value.shape == ()
        assert np.isfinite(value.numpy())


def test_vae_trains_one_step(synthetic_batch: tf.Tensor) -> None:
    """The VAE compiles and runs one training step without error."""
    vae = VAE(build_vae_encoder(), build_vae_decoder(), kl_weight=0.25)
    vae.compile(optimizer="adam")
    history = vae.fit(
        synthetic_batch,
        synthetic_batch,
        epochs=1,
        batch_size=BATCH_SIZE_FOR_TEST,
        verbose=0,
    )
    for key in ("loss", "reconstruction_loss", "kl_loss"):
        assert key in history.history
        assert np.isfinite(history.history[key][0])


def test_vae_kl_weight_is_stored() -> None:
    """The constructor records ``kl_weight`` for use during training."""
    vae = VAE(build_vae_encoder(), build_vae_decoder(), kl_weight=0.25)
    assert vae.kl_weight == 0.25
