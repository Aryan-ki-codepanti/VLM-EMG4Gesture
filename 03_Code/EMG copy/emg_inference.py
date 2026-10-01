from pathlib import Path
import sys

import numpy as np
import torch
from scipy.io import loadmat

from .data_preparation import EMGDataPreparation
from .preprocessing import EMGPreprocessing

from Prompt.mappings import EMG_LABEL_MAP


def load_emg_array(file_path):
    """
    Load a single EMG MAT file.
    Return shape: (samples, 8).
    """
    file_path = Path(file_path)
    data = loadmat(file_path)

    candidates = [
        value
        for key, value in data.items()
        if not key.startswith("__")
        and isinstance(value, np.ndarray)
        and value.ndim == 2
    ]

    if not candidates:
        raise ValueError(
            f"No 2D EMG array found in {file_path}"
        )

    emg = np.asarray(candidates[0], dtype=np.float32)

    # Convert to (samples, channels)
    if emg.shape[0] == 8:
        emg = emg.T

    if emg.shape[1] != 8:
        raise ValueError(
            f"Expected 8 channels, got shape {emg.shape}"
        )

    return emg


def preprocess_emg(
    emg,
    fs=200,
    notch_freq=60.0,
    low_cut=10.0,
    high_cut=99.0,
    order=5,
    window_time=200,
    overlap=60,
    no_channel=8
):
    """
    Reuse the original EMG preprocessing and windowing pipeline.

    Input:
        emg: (samples, channels)

    Output:
        windows: (num_windows, 1, channels, window_samples)
    """

    prep = EMGDataPreparation(
        base_path=".",
        fs=fs,
        rec_time=5
    )

    processor = EMGPreprocessing(
        fs=fs,
        notch_freq=notch_freq,
        low_cut=low_cut,
        high_cut=high_cut,
        order=order
    )

    # Original repository expects (channels, samples)
    emg = emg.T

    # Filtering along the samples axis
    emg = processor.remove_mains(emg)
    emg = processor.bandpass_filter(emg)

    # One label per sample for windowing
    dummy_labels = np.zeros(
        emg.shape[1],
        dtype=np.int64
    )

    windows, _ = prep.window_with_overlap(
        emg,
        dummy_labels,
        window_time=window_time,
        overlap=overlap,
        no_channel=no_channel
    )

    # Model expects (batch, 1, channels, window_samples)
    windows = np.expand_dims(windows, axis=1)

    return windows.astype(np.float32)


def infer_emg(file_path, model, device):
    """
    Run EMG inference on the first available window.
    """

    emg = load_emg_array(file_path)

    windows = preprocess_emg(emg)

    if len(windows) == 0:
        raise ValueError(
            f"No valid EMG windows generated for {file_path}"
        )

    tensor = torch.tensor(
        windows,
        dtype=torch.float32
    ).to(device)

    model.eval()

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)

    # Average probabilities over all windows
    mean_probabilities = probabilities.mean(dim=0)

    confidence, predicted_class = torch.max(
        mean_probabilities,
        dim=0
    )

    predicted_class = predicted_class.item()
    confidence = confidence.item()

    return {
        "gesture": EMG_LABEL_MAP[predicted_class],
        "class_id": predicted_class,
        "confidence": round(confidence, 4),
        "num_windows": len(windows),
        "file": str(file_path)
    }
