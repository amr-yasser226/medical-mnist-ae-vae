"""AE and beta-VAE model definitions.

Models follow the Keras functional API style introduced in Lab 3:
    - Conv2D + MaxPooling2D encoders
    - Conv2DTranspose decoders
    - Sigmoid output heads
    - Pixelwise binary cross-entropy reconstruction loss

The VAE is implemented as a custom ``keras.Model`` subclass with a
``train_step`` override that accumulates reconstruction and KL terms
separately for diagnostic plotting.
"""

from __future__ import annotations

from typing import Tuple

import keras
import tensorflow as tf
from keras import layers

from src.data_processing import IMG_SIZE

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
LATENT_DIM: int = 16
AE_BOTTLENECK_CHANNELS: int = 16


# ---------------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------------
def build_autoencoder() -> Tuple[keras.Model, keras.Model]:
    """Build a convolutional autoencoder for ``IMG_SIZE x IMG_SIZE`` images.

    Encoder: three ``Conv2D + MaxPooling2D`` blocks downsampling
    ``(IMG_SIZE, IMG_SIZE, 1)`` to ``(8, 8, AE_BOTTLENECK_CHANNELS)``.
    Decoder: three ``Conv2DTranspose`` blocks upsampling back to the input
    resolution, with a final 3x3 sigmoid ``Conv2D`` to constrain outputs to
    ``[0, 1]``.

    Returns:
        Tuple ``(autoencoder, encoder)``. The encoder shares weights with the
        first half of the autoencoder and is exposed for latent-space
        visualisation.
    """
    inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inp)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)              # 32x32
    x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)              # 16x16
    x = layers.Conv2D(
        AE_BOTTLENECK_CHANNELS, (3, 3), activation="relu", padding="same"
    )(x)
    encoded = layers.MaxPooling2D((2, 2), padding="same")(x)        # 8x8xC

    x = layers.Conv2DTranspose(
        16, (3, 3), strides=(2, 2), activation="relu", padding="same"
    )(encoded)
    x = layers.Conv2DTranspose(
        32, (3, 3), strides=(2, 2), activation="relu", padding="same"
    )(x)
    x = layers.Conv2DTranspose(
        32, (3, 3), strides=(2, 2), activation="relu", padding="same"
    )(x)
    decoded = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

    autoencoder = keras.Model(inp, decoded, name="autoencoder")
    encoder = keras.Model(inp, encoded, name="ae_encoder")
    return autoencoder, encoder


# ---------------------------------------------------------------------------
# Variational Autoencoder
# ---------------------------------------------------------------------------
class Sampling(layers.Layer):
    """Reparameterisation trick layer.

    Given ``(z_mean, z_log_var)`` returns ``z = mu + sigma * epsilon`` with
    ``epsilon ~ N(0, I)``, allowing gradients to flow through the stochastic
    sampling step.
    """

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_vae_encoder(latent_dim: int = LATENT_DIM) -> keras.Model:
    """Build the VAE encoder mapping inputs to ``(z_mean, z_log_var, z)``.

    Args:
        latent_dim: Dimensionality of the latent code.

    Returns:
        ``keras.Model`` with three outputs: posterior mean, posterior log
        variance, and a reparameterised sample from the posterior.
    """
    inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = layers.Conv2D(32, 3, strides=2, activation="relu", padding="same")(inp)
    x = layers.Conv2D(64, 3, strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(64, 3, strides=2, activation="relu", padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    return keras.Model(inp, [z_mean, z_log_var, z], name="vae_encoder")


def build_vae_decoder(latent_dim: int = LATENT_DIM) -> keras.Model:
    """Build the VAE decoder mapping a latent vector to a reconstruction.

    Args:
        latent_dim: Dimensionality of the latent code.

    Returns:
        ``keras.Model`` mapping ``(latent_dim,)`` to
        ``(IMG_SIZE, IMG_SIZE, 1)``.
    """
    inp = keras.Input(shape=(latent_dim,))
    x = layers.Dense(8 * 8 * 64, activation="relu")(inp)
    x = layers.Reshape((8, 8, 64))(x)
    x = layers.Conv2DTranspose(
        64, 3, strides=2, activation="relu", padding="same"
    )(x)
    x = layers.Conv2DTranspose(
        32, 3, strides=2, activation="relu", padding="same"
    )(x)
    x = layers.Conv2DTranspose(
        32, 3, strides=2, activation="relu", padding="same"
    )(x)
    out = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)
    return keras.Model(inp, out, name="vae_decoder")


class VAE(keras.Model):
    """beta-Variational Autoencoder.

    Combines a probabilistic encoder, the reparameterisation trick, and a
    decoder. The training objective is the evidence lower bound with a scalar
    KL weight ``beta``::

        L = E_q[BCE(x, x_hat)] + beta * KL(q(z | x) || N(0, I))

    With ``beta = 1`` this is the standard VAE. With ``beta < 1`` it is the
    beta-VAE variant favouring reconstruction sharpness; with ``beta > 1`` it
    favours disentanglement at the cost of reconstruction.
    """

    def __init__(
        self,
        encoder: keras.Model,
        decoder: keras.Model,
        kl_weight: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.kl_weight = kl_weight
        self.total_tracker = keras.metrics.Mean(name="loss")
        self.rec_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_tracker, self.rec_tracker, self.kl_tracker]

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        _, _, z = self.encoder(inputs, training=training)
        return self.decoder(z, training=training)

    def _compute_losses(
        self,
        x: tf.Tensor,
        x_hat: tf.Tensor,
        z_mean: tf.Tensor,
        z_log_var: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Return ``(total, reconstruction, kl)`` losses as scalar tensors."""
        bce = keras.losses.binary_crossentropy(x, x_hat)
        # Sum over spatial dims, mean over batch -- standard for VAEs.
        rec = tf.reduce_mean(tf.reduce_sum(bce, axis=(1, 2)))
        kl = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                axis=1,
            )
        )
        total = rec + self.kl_weight * kl
        return total, rec, kl

    def train_step(self, data):
        x, _ = data if isinstance(data, tuple) else (data, data)
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x, training=True)
            x_hat = self.decoder(z, training=True)
            total, rec, kl = self._compute_losses(x, x_hat, z_mean, z_log_var)
        grads = tape.gradient(total, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_tracker.update_state(total)
        self.rec_tracker.update_state(rec)
        self.kl_tracker.update_state(kl)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, _ = data if isinstance(data, tuple) else (data, data)
        z_mean, z_log_var, z = self.encoder(x, training=False)
        x_hat = self.decoder(z, training=False)
        total, rec, kl = self._compute_losses(x, x_hat, z_mean, z_log_var)
        self.total_tracker.update_state(total)
        self.rec_tracker.update_state(rec)
        self.kl_tracker.update_state(kl)
        return {m.name: m.result() for m in self.metrics}
