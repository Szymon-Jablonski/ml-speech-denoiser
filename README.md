# Speech Audio Denoiser

## Project Description

Speech Audio Denoiser is a university machine learning project about reducing background noise in speech recordings. The project will use PyTorch to train a model that takes a noisy speech recording as input and tries to produce a clearer version.

## Goal

The goal is to learn how a neural network can separate speech from unwanted sounds, such as steady background noise. The model's output will be compared with the original clean recording to see how well it works, as well as compared with other denoising models and techniques. This project is for learning and experimentation; it is not intended to remove every kind of noise perfectly.

## Planned Tools and Methods

- **Python** for the project code.
- **PyTorch** for building, training, and evaluating the machine learning model.
- **Audio and signal-processing libraries** such as `torchaudio` for loading, preparing, and listening to recordings.
- **Paired audio examples**: noisy recordings together with matching clean speech recordings, used to teach the model what the result should sound like.
- **Spectrograms** to represent audio in a way that shows how its frequencies change over time. The model may work with these representations, depending on the final approach.
- **Evaluation and listening tests** to compare the noisy input with the model's output and the clean reference.

## Basic Project Plan

1. Find or prepare a dataset containing clean speech and matching noisy speech.
2. Load the recordings and prepare them for training.
3. Build and train a PyTorch model to reduce noise.
4. Test the model on recordings it has not seen during training.
5. Compare the results using simple measurements and by listening to the audio.

## Expected Result

The project should produce a working experiment that can take a noisy speech recording and generate a denoised version. It will also show what the model does well, where it struggles, and how the quality of the training data affects the result.

## Denoising Approaches

This project will investigate two ways of training a neural network for speech denoising:

1. **Supervised denoising** – train the model using noisy recordings as inputs and their matching clean recordings as targets.
2. **Self-supervised Noise2Noise denoising** – train the model using two independently corrupted versions of the same clean recording, without using the clean recording as the training target.

Both approaches can use the same neural network architecture. Comparing them with the same data splits, preprocessing, and evaluation procedure makes it easier to study the effect of the training objective itself.

### 1. Supervised denoising

In supervised learning, the model is shown both the noisy audio and the clean audio that it should learn to reproduce.

#### Preparing the training examples

Start with a clean speech recording $x(t)$ and a noise recording $n(t)$. Mix them to create a noisy recording:

```math
y(t) = x(t) + \alpha \, n(t)
```

Here, $\alpha$ controls the noise level. It can be varied to create examples with different signal-to-noise ratios (SNRs). Before mixing, make sure the speech and noise have the same sample rate and compatible lengths. Noise can be cropped or looped as needed, and its level can be adjusted to achieve a chosen SNR.

Each training example consists of:

- **Input:** the noisy recording $y(t)$.
- **Target:** the corresponding clean recording $x(t)$.

#### Using spectrograms

A practical first implementation is to work in the frequency domain:

1. Load the clean speech and noise audio.
2. Create a noisy mixture.
3. Compute the Short-Time Fourier Transform (STFT) of the noisy mixture and clean speech.
4. Feed the noisy spectrogram to the network.
5. Train the network to predict the clean spectrogram.

In notation:

```math
Y = \mathrm{STFT}(y), \qquad X = \mathrm{STFT}(x)
```

```math
\hat{X} = f_\theta(Y)
```

The network $f_\theta$ can initially be a relatively small convolutional neural network (CNN) or U-Net. A simple starting loss is the mean absolute error (L1):

```math
\mathcal{L}_{\mathrm{sup}} = \lVert \hat{X} - X \rVert_1
```

Mean squared error (MSE) is another option. The initial implementation can predict the magnitude spectrogram, which is simpler to handle. To reconstruct audio from a predicted magnitude spectrogram, the phase of the noisy input is commonly reused; this is an approximation and can limit audio quality. A later experiment could predict the real and imaginary components of the complex STFT instead.

Finally, convert the predicted representation back to a waveform using the inverse STFT.

#### Implementation outline

1. Create a dataset that returns aligned `(noisy_spectrogram, clean_spectrogram)` pairs.
2. Split the original clean recordings and noise recordings into training, validation, and test sets **before** creating mixtures. This helps prevent closely related mixtures from leaking across splits.
3. Implement a CNN or U-Net that accepts the chosen spectrogram representation and predicts the clean representation.
4. Train using the supervised loss, monitoring validation loss and audio quality.
5. Save the best model checkpoint and test it on recordings that were not used for training or model selection.

### 2. Self-supervised denoising with Noise2Noise

Noise2Noise uses a different training target. Instead of requiring a clean target for every input, it learns from two independently corrupted versions of the same underlying clean recording.

#### Preparing the training examples

Take one clean recording $x(t)$ and create two noisy versions using independent noise realizations:

```math
y_1(t) = x(t) + n_1(t)
```

```math
y_2(t) = x(t) + n_2(t)
```

The noise signals $n_1(t)$ and $n_2(t)$ should be independently sampled and, for the standard Noise2Noise justification, have zero mean and be independent of the clean signal.

Each training example consists of:

- **Input:** the first noisy version $y_1(t)$.
- **Target:** the second noisy version $y_2(t)$.

The clean recording is used to create the two views, but it is **not** used as the target in the Noise2Noise training loss.

#### Training the model

Compute the STFT of both noisy versions:

```math
Y_1 = \operatorname{STFT}(y_1), \qquad Y_2 = \operatorname{STFT}(y_2)
```

The network receives the first noisy spectrogram and predicts the second:

```math
\hat{Y}_2 = f_\theta(Y_1)
```

A simple loss is:

```math
\mathcal{L}_{\mathrm{N2N}} = \lVert \hat{Y}_2 - Y_2 \rVert_2^2
```

The intuition is that the two noise realizations are unrelated. Across many examples, predicting the noise in the second view does not help the model minimize the expected squared error; predicting the shared underlying speech does. Under the assumptions above, the optimal prediction is therefore the clean signal representation.

For the standard mathematical argument, it is safest to apply the loss to the **complex STFT** (or to the waveform). The STFT is linear, so additive zero-mean noise remains zero-mean in this representation. Magnitude is a nonlinear operation, however, and the magnitude of a noisy spectrogram is not simply the clean magnitude plus zero-mean noise. Therefore, the simple Noise2Noise guarantee does not directly carry over to a magnitude-only target.

#### Implementation outline

1. Start with a clean training recording and generate two mixtures using independently selected noise segments or independently generated noise.
2. Compute both complex STFTs.
3. Create a dataset that returns `(noisy_view_1, noisy_view_2)` pairs. Do not return the clean spectrogram as the training target.
4. Use the same CNN or U-Net architecture as in the supervised experiment, adapted to accept and predict the real and imaginary STFT components.
5. Train with the Noise2Noise loss and monitor validation performance.
6. At inference time, provide just one noisy recording to the model. The model does not need a second noisy version.

> [!IMPORTANT]
> This is self-supervised training because the target is another noisy view rather than a clean label. If both views are synthesized from clean recordings, however, the experiment still relies on access to clean source recordings to construct its training examples. It does not yet demonstrate learning from naturally noisy, unlabelled recordings alone.

### 3. Suggested experiment workflow

A manageable order for implementing the project is:

1. **Prepare and inspect the data.** Load clean speech and noise recordings, check sample rates and durations, and listen to the source recordings.
2. **Create leakage-safe splits.** Split source recordings into training, validation, and test groups before generating noisy mixtures.
3. **Build a shared preprocessing pipeline.** Implement mixing, SNR control, STFT, and inverse STFT. Check that audio can be reconstructed correctly when no denoising is applied.
4. **Implement the supervised baseline.** Train a small CNN or U-Net using noisy-to-clean pairs.
5. **Implement Noise2Noise.** Reuse the architecture and preprocessing where possible, but create independent noisy views and train from view 1 to view 2.
6. **Evaluate both models consistently.** Use the same held-out test mixtures and clean references, calculate objective metrics, and listen to the outputs.
7. **Discuss limitations.** Consider the effects of noise type, SNR, training data size, phase handling, and the assumptions required by Noise2Noise.

## How To

The project is a Jupyter notebook ([`audio_denoiser.ipynb`](audio_denoiser.ipynb)) that is meant to run in **Google Colab**, so it can use Colab's free GPU instead of local hardware.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Szymon-Jablonski/ml-audio-denoiser/blob/main/audio_denoiser.ipynb)

### Run the notebook in Google Colab

1. **Open the notebook.** Click the *Open in Colab* badge above, or in Colab go to *File → Open notebook → GitHub*, paste `https://github.com/Szymon-Jablonski/ml-audio-denoiser` and choose `audio_denoiser.ipynb`.
2. **Turn on the GPU.** Go to *Runtime → Change runtime type*, select *T4 GPU* (or any available GPU) and click *Save*.
3. **Run the cells.** Run them from top to bottom (*Runtime → Run all*, or `Shift + Enter` cell by cell). The first cell installs the required packages and the next one prints whether the GPU is visible.
4. **(Optional) Mount Google Drive.** Colab's disk is wiped when the runtime disconnects. Set `USE_DRIVE = True` in the configuration cell to store datasets and model checkpoints in your Google Drive (`MyDrive/ml-audio-denoiser/`) so they survive between sessions.
5. **Save your changes.** Edits made in Colab are not written to this repository automatically. Use *File → Save a copy in GitHub*, or download the notebook (*File → Download → .ipynb*) and commit it.

### Running locally (optional)

The notebook can also run on a local machine (training will be slow without a GPU):

```bash
pip install torch torchaudio soundfile matplotlib jupyter
jupyter notebook audio_denoiser.ipynb
```

### Repository structure

- `audio_denoiser.ipynb` – main notebook: setup, data preparation, model, training and evaluation.
- `README.md` – project description and instructions.
