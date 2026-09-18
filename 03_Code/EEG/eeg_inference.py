
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

from Prompt.mappings import EEG_GESTURE_MAP


def load_eeg_array(file_path):
    """
    Load EEG data from .mat or .csv and normalize its shape
    to (8, 1250).
    """
    file_path = Path(file_path)

    if file_path.suffix.lower() == ".mat":
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
                f"No 2D EEG array found in {file_path}"
            )

        eeg = candidates[0]

    elif file_path.suffix.lower() == ".csv":
        eeg = np.loadtxt(file_path, delimiter=",")

    else:
        raise ValueError(
            f"Unsupported EEG file format: {file_path.suffix}"
        )

    eeg = np.asarray(eeg, dtype=np.float32)

    if eeg.shape == (1250, 8):
        eeg = eeg.T

    if eeg.shape != (8, 1250):
        raise ValueError(
            f"Expected EEG shape (8, 1250), got {eeg.shape}"
        )

    return eeg


def normalize_eeg(eeg):
    """
    Apply per-channel z-score normalization.
    """
    eeg = eeg.astype(np.float32)

    mean = eeg.mean(axis=1, keepdims=True)
    std = eeg.std(axis=1, keepdims=True)

    return (eeg - mean) / (std + 1e-8)


def infer_eeg(file_path, model, device):
    """
    Run EEG inference and return gesture + confidence.
    """
    eeg = load_eeg_array(file_path)
    eeg = normalize_eeg(eeg)

    tensor = torch.tensor(
        eeg,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)

    confidence, predicted_class = torch.max(
        probabilities,
        dim=1
    )

    predicted_class = predicted_class.item()
    confidence = confidence.item()

    gesture_key = f"G{predicted_class + 1}"

    if gesture_key not in EEG_GESTURE_MAP:
        raise KeyError(
            f"Unknown EEG gesture key: {gesture_key}"
        )

    return {
        "gesture": EEG_GESTURE_MAP[gesture_key],
        "class_id": predicted_class,
        "confidence": round(confidence, 4),
        "file": str(file_path)
    }
