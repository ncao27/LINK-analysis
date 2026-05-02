# LINK-NSP Project 🌐

This project explores neural decoding using the **LINK dataset**, which contains neural recordings from a macaque performing dexterous finger movements. The goal is to predict **finger position** from neural activity (e.g., Spiking Band Power) using machine learning models, particularly an **LSTM neural network**.

The repository is structured so that preprocessing, dataset construction, model definition, and training logic are separated into modular files.

## Setup

Create the environment:

```bash
conda env create -f environment.yml
conda activate neurocuda
```

---

# Repository Structure

```
LINK-NSP
│
├── data/
│
├── models/
│   └── lstm.py
│
├── src/
│   ├── preprocessing.py
│   ├── training.py
│   └── window.py
│
├── notebooks/
│   └── link_analysis.ipynb
│
└── README.md
```


---

# Notebook

### `notebooks/link_analysis.ipynb`

The notebook is used for running experiments and interacting with the dataset. It loads sessions, applies preprocessing, constructs datasets, trains models, and visualizes results. The core logic for these steps lives in the `src/` modules.

---

# Models

## `models/lstm.py`

Defines the LSTM-based neural decoder used to predict continuous finger position.

### `LSTMDecoder`

An LSTM regression model that takes neural activity sequences as input and outputs predicted finger position.

Input shape: (batch_size, sequence_length, num_channels)
Example: (128, 100, 96)

Components:

- **LSTM layer** – processes temporal neural activity
- **Linear output layer** – maps the final hidden state to a scalar prediction

### `forward(x)`

Defines the forward pass:

1. Pass sequence through the LSTM
2. Take the hidden state at the final time step
3. Pass it through a linear layer to produce the prediction

### `time_split(X, y, train_frac=0.8)`

Splits time-series data into training and testing segments using a sequential split rather than random shuffling.

---

# Source Modules

All reusable functionality lives in the `src/` directory.

---

# `src/window.py`

Creates datasets suitable for LSTM training using sliding windows.

### `NeuralSequenceDataset`

Converts continuous neural recordings into sequences.

Input data format: (time, channels)

Each dataset sample becomes: (sequence_length, channels)


paired with a target finger position.

Functions:

- **`__len__()`** – returns number of valid sliding windows
- **`__getitem__(idx)`** – returns `(x_seq, y_target)` for a given index

---

# `src/preprocessing.py`

Contains neural signal preprocessing functions. All functions assume data format: (time, channels)


Functions:

### `car(neural_data)`

Common Average Referencing (CAR). Removes shared noise across electrodes by subtracting the average signal across channels.

### `bandpass(fs, low_fc, high_fc, neural_data, order=4)`

Applies a Butterworth bandpass filter using zero-phase filtering (`filtfilt`).

### `whitening(neural_data)`

Performs channel whitening by removing cross-channel correlations using the covariance matrix.

### `thres_det(neural_data, threshold_multiplier=-4.5)`

Detects negative threshold crossings used for spike detection.

### `smoothing(neural_data, sigma=2)`

Applies Gaussian smoothing across time to reduce noise.

### `zscore_channels(neural_data)`

Normalizes neural signals per channel using z-score normalization.

### `apply_pca(neural_data, n_components=10)`

Performs dimensionality reduction using PCA and returns both the reduced data and the PCA model.

---

# `src/training.py`

Contains functions for training and evaluating the LSTM decoder.

### `train_lstm_model(...)`

Trains the LSTM model using mini-batch gradient descent.

Training loop steps:

1. Set model to training mode
2. Iterate over batches from `train_loader`
3. Compute predictions
4. Compute MSE loss
5. Backpropagate gradients
6. Update weights using Adam optimizer

Progress is printed periodically during training.

Outputs:

- training loss history
- test R² scores

### `evaluate_lstm_model(model, data_loader, device="cpu")`

Evaluates model performance on a dataset.

Steps:

1. Run inference on batches
2. Collect predictions and ground truth
3. Compute **R² score** using `sklearn.metrics.r2_score`

---

# Summary

The project implements a neural decoding pipeline consisting of:

1. Neural preprocessing
2. Sliding-window sequence construction
3. LSTM-based neural decoding
4. Training and evaluation using R² metrics

The modular structure allows each component of the pipeline to be reused and extended for additional experiments.









