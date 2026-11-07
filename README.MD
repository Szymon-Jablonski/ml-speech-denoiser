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
- `README.MD` – project description and instructions.
