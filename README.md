# Representation Learning with Autoencoders (AE & VAE)

Convolutional Autoencoder (AE) and Variational Autoencoder (VAE) trained on the [Medical MNIST](https://www.kaggle.com/datasets/andrewmvd/medical-mnist) dataset (six classes of 64x64 grayscale medical images: AbdomenCT, BreastMRI, CXR, ChestCT, Hand, HeadCT).

Covers reconstruction, latent-space visualisation, sampling from the VAE prior, latent interpolation, and denoising — and compares the two architectures.

## Repository layout

```
.
├── notebooks/
│   └── ae_vae_medical_mnist.ipynb   # Kaggle notebook (primary deliverable)
├── src/                              # Modular Python source code (data pipeline, models, training)
├── tests/                            # Unit tests for data and model logic
├── data/                             # Dataset storage (e.g. data/raw/medicalMNIST)
├── models/                           # Saved Keras model weights
├── figures/                          # Figures produced by the notebook or training script
├── report/
│   ├── report.tex                    # 2-page technical report source
│   └── report.pdf                    # compiled report (produced from report.tex)
├── presentation/
│   ├── slides.tex                    # Beamer slides for the video demo
│   └── slides.pdf                    # compiled slides
├── demo.mkv                          # Video demonstration recording
├── requirements.txt
└── README.md
```

## How to run (Local Python Script)

The repository now includes a modularized Python package for local training.

1. Unzip the included dataset archive: `unzip data/raw/medicalMNIST.zip -d data/raw/medical-mnist/`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the end-to-end training script:
   ```bash
   python -m src.train --data-root data/raw/medical-mnist --epochs 30
   ```
4. Run unit tests:
   ```bash
   pytest tests/ -v
   ```

## How to run (Kaggle)

1. Create a new Kaggle notebook.
2. Click **Add Input** and attach the Medical MNIST dataset by andrewmvd.
3. Upload `notebooks/ae_vae_medical_mnist.ipynb`.
4. Run all cells.

Outputs land under `/kaggle/working/`:
- `figures/` — all eight PNG figures (dataset samples, AE loss, VAE loss, reconstruction comparison, latent-space PCA, VAE prior samples, latent interpolation, denoising)
- `models/` — saved AE and VAE weights (`ae.keras`, `vae_encoder.keras`, `vae_decoder.keras`)
- `metrics.txt` — test-set MSE and MAE for both models

Download `figures/` from `/kaggle/working/` into the local `figures/` folder of this repo, then compile the report and slides.

## Models at a glance

- **AE**: `Conv2D + MaxPooling2D` encoder → 8×8×8 bottleneck (512 dims) → `Conv2DTranspose` decoder → sigmoid output. Loss: pixelwise BCE.
- **VAE**: strided `Conv2D` encoder → 16-dim `(mu, log_var)` → reparameterised sample → `Conv2DTranspose` decoder. Loss: pixel-summed BCE + KL to N(0, I).

Both follow the Keras functional-API style used in Lab 3.

## Deliverables mapping

| Assignment requirement           | Location                                                  |
|----------------------------------|-----------------------------------------------------------|
| AE implementation                | notebook §3                                               |
| VAE implementation               | notebook §4                                               |
| Latent-space visualisation       | notebook §6, `figures/05_latent_space_pca.png`            |
| Reconstruction analysis          | notebook §5, `figures/04_reconstruction_comparison.png`   |
| Sample generation                | notebook §7, `figures/06_vae_generated_samples.png`, `figures/07_vae_interpolation.png` |
| Denoising                        | notebook §8, `figures/08_denoising.png`                   |
| Loss visualisation               | `figures/02_ae_loss.png`, `figures/03_vae_loss.png`       |
| `tf.data` pipeline               | notebook §2                                               |
| Technical report (2 pages)       | `report/report.pdf`                                       |
| Video demonstration              | `demo.mkv` (Note: The audio volume is unexpectedly low)   |

## Honest notes on expected results

- Medical MNIST is a very easy dataset. Reconstructions from both models look nearly identical to the originals; this is the dataset being easy, not the models being remarkable.
- VAE samples from N(0, I) are recognisably medical-image-shaped but blurry and sometimes class-ambiguous. This is the standard blurriness of a small convolutional VAE and is discussed in the report.
- The AE latent space is better separated by class than the VAE latent space because the VAE's KL regulariser pulls all classes toward the same prior. This is the expected trade-off (class separability vs. ability to sample), not a bug.

## Building the report and slides

```bash
cd report       && pdflatex report.tex && pdflatex report.tex && pdflatex report.tex
cd ../presentation && pdflatex slides.tex && pdflatex slides.tex
```

Three passes are needed for the report so cross-references stabilise.
