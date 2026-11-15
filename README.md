# Speech Audio Denoiser

A university machine learning project that uses PyTorch to reduce background noise in speech recordings. A model takes a noisy recording and tries to produce a cleaner version. The result is compared with the clean reference and with other denoising methods. The goal is to learn and experiment, not to remove every kind of noise perfectly.

**Tools:** Python, PyTorch, `torchaudio`, spectrograms (STFT), objective metrics and listening tests.

**Data:** clean Polish speech from [Common Voice](https://commonvoice.mozilla.org/) mixed with environmental noise from [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html).

## Denoising Approaches

We compare two training objectives. Both use the same network architecture, data splits and evaluation, so any difference in results comes from the objective itself.

### 1. Supervised (noisy → clean)

Mix clean speech $x(t)$ with noise $n(t)$, where $\alpha$ sets the signal-to-noise ratio (SNR):

```math
y(t) = x(t) + \alpha \, n(t)
```

The network $f_\theta$ (a small CNN or U-Net) maps the noisy spectrogram to the clean one:

```math
Y = \mathrm{STFT}(y), \quad X = \mathrm{STFT}(x), \quad \hat{X} = f_\theta(Y), \quad \mathcal{L}_{\mathrm{sup}} = \lVert \hat{X} - X \rVert_1
```

The first version predicts magnitude spectrograms. To turn the output back into audio, it reuses the phase of the noisy input, which is an approximation. A later version can predict the complex STFT (real and imaginary parts) instead. Clean and noise recordings are split into train, validation and test sets **before** mixing, so related mixtures cannot leak across splits.

### 2. Self-supervised Noise2Noise (noisy → noisy)

Make two noisy versions of the same clean recording, each with its own independently sampled zero-mean noise:

```math
y_1(t) = x(t) + n_1(t), \qquad y_2(t) = x(t) + n_2(t)
```

```math
\hat{Y}_2 = f_\theta(Y_1), \qquad \mathcal{L}_{\mathrm{N2N}} = \lVert \hat{Y}_2 - Y_2 \rVert_2^2
```

The two noise signals are unrelated, so the model cannot learn to predict $n_2$. The prediction that minimises the expected error is therefore the clean speech. This argument holds for the **complex STFT** or the waveform, because both are linear in the signal. It does not hold for magnitude spectrograms. At inference time the model needs only one noisy recording.

> [!IMPORTANT]
> Both noisy versions are still made from clean recordings. This shows the training objective works, but it does not show learning from naturally noisy, unlabelled data.

### Workflow

1. Inspect the data and create leakage-safe splits.
2. Build a shared pipeline: mixing, SNR control, STFT and inverse STFT. Check that audio comes back unchanged when no denoising is applied.
3. Train the supervised baseline, then Noise2Noise with the same architecture.
4. Evaluate both on the same held-out mixtures, using metrics and listening.
5. Discuss limitations: noise type, SNR, data size, phase handling and the Noise2Noise assumptions.

## Running in Google Colab

The notebooks are designed for Google Colab, so training can use its free GPU.

| Notebook | Open |
|---|---|
| `denoiser_generate_datasets.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Szymon-Jablonski/ml-speech-denoiser/blob/main/denoiser_generate_datasets.ipynb) |
| `denoiser_cnn.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Szymon-Jablonski/ml-speech-denoiser/blob/main/denoiser_cnn.ipynb) |
| `denoiser_noise2noise.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Szymon-Jablonski/ml-speech-denoiser/blob/main/denoiser_noise2noise.ipynb) |

### 1. Add the shared data folder to your Drive

The datasets are stored in a shared Google Drive folder:
https://drive.google.com/drive/folders/1FogEgAKr39y7ODDGQTP3SFny2-JZ5sPK?usp=sharing

The notebooks expect the data at `MyDrive/Denoiser/`. To set this up:

1. Open the link above, right-click the folder and choose **Organize → Add shortcut**. Pick **My Drive** as the location.
2. Make sure the shortcut is named **`Denoiser`**. It should contain `CommonVoice27_PL.tar`, `UrbanSound8K.tar` and `DenoiserData.tar`.

### 2. Run a notebook

1. Open the notebook with one of the badges above. You can also go to *File → Open notebook → GitHub* in Colab and paste the repository URL.
2. Select *Runtime → Change runtime type → T4 GPU*.
3. Run the first cell. It mounts your Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
   Allow access when prompted. The shared files then appear under `/content/drive/MyDrive/Denoiser/`.
4. Run the remaining cells from top to bottom. They unpack the `.tar` archives to `/content/`, which is Colab's fast local disk, and install the required packages.

Colab's local disk is wiped when the runtime disconnects. Keep anything you want to reuse, such as datasets or checkpoints, on Drive. Changes you make in Colab are not saved to this repository. Use *File → Save a copy in GitHub*, or download the `.ipynb` file and commit it yourself.

### Running locally (optional)

To run locally, put the extracted datasets in the repository root and replace the `/content/...` paths in the notebooks with your local paths. Skip the `drive.mount` cell.

```bash
pip install torch torchaudio soundfile matplotlib pandas tqdm jupyter
jupyter notebook
```

## Repository Structure

- [`denoiser_generate_datasets.ipynb`](denoiser_generate_datasets.ipynb): builds the training and validation sets. It mixes Common Voice speech with UrbanSound8K noise at a chosen SNR and saves the result as `DenoiserData.tar`.
- [`denoiser_cnn.ipynb`](denoiser_cnn.ipynb): the supervised CNN denoiser. It loads `DenoiserData`, generates spectrograms, and covers the model, training and evaluation.
- [`denoiser_noise2noise.ipynb`](denoiser_noise2noise.ipynb): the self-supervised Noise2Noise denoiser (work in progress).
- `README.md`: project description and instructions.
