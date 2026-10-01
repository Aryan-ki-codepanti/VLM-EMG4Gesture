
import numpy as np
import pandas as pd
import scipy.io as sio
import os
import re

# --- Configuration based on Design Blueprint and Audit ---
DEFAULT_SFREQ = 200  # Hz - Assumed sampling frequency

def load_eeg_mat(filepath):
    """
    Loads EEG data from a .mat file.
    Assumes data is under the key 'data' and is shaped (n_channels, n_samples).
    """
    mat_data = sio.loadmat(filepath)
    data = mat_data.get('data') # Get the 'data' key

    if data is None:
        raise ValueError(f"'data' key not found in {filepath}. Available keys: {mat_data.keys()}")

    if not isinstance(data, np.ndarray):
        raise TypeError(f"Data loaded from {filepath} is not a NumPy array. Type: {type(data)}")

    filename = os.path.basename(filepath)
    filename_match = re.search(r'S(\d+)_R(\d+)_G(\d+)', filename)
    metadata = {}
    if filename_match:
        metadata['subject'] = int(filename_match.group(1))
        metadata['run'] = int(filename_match.group(2))
        metadata['gesture'] = int(filename_match.group(3))

    metadata['sfreq'] = DEFAULT_SFREQ # Attach assumed sampling frequency

    return data, metadata

def load_eeg_csv(filepath):
    """
    Loads EEG data from a .csv file.
    Assumes the CSV, once loaded by pandas, is already in (n_channels, n_samples) format.
    """
    df = pd.read_csv(filepath)
    data = df.values.astype(np.float64)

    filename = os.path.basename(filepath)
    filename_match = re.search(r'S(\d+)_R(\d+)_G(\d+)', filename)
    metadata = {}
    if filename_match:
        metadata['subject'] = int(filename_match.group(1))
        metadata['run'] = int(filename_match.group(2))
        metadata['gesture'] = int(filename_match.group(3))

    metadata['sfreq'] = DEFAULT_SFREQ # Attach assumed sampling frequency

    return data, metadata

def load_eeg_data(filepath):
    """
    Generic function to load EEG data based on file extension.
    Returns a tuple (data_array, metadata_dict) where data_array is (n_channels, n_samples).
    """
    if filepath.lower().endswith('.mat'):
        return load_eeg_mat(filepath)
    elif filepath.lower().endswith('.csv'):
        return load_eeg_csv(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}. Only .mat and .csv are supported.")
