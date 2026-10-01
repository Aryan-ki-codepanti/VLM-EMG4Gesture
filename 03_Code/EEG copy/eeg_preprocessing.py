
import mne
from sklearn.preprocessing import StandardScaler
import numpy as np # For type hinting and array operations
from eeg_data_preparation import DEFAULT_SFREQ # Import DEFAULT_SFREQ with absolute path

def preprocess_eeg_data(eeg_data, sfreq=DEFAULT_SFREQ, target_sfreq=DEFAULT_SFREQ, l_freq=1, h_freq=30, notch_freqs=50):
    """
    Applies preprocessing steps to raw EEG data.

    Args:
        eeg_data (np.ndarray): EEG data array with shape (n_channels, n_samples).
        sfreq (float): Original sampling frequency of the EEG data.
        target_sfreq (float): Target sampling frequency for resampling.
        l_freq (float): Lower cutoff frequency for bandpass filter (Hz).
        h_freq (float): Higher cutoff frequency for bandpass filter (Hz).
        notch_freqs (float or list of floats): Frequencies to notch filter (Hz).

    Returns:
        np.ndarray: Preprocessed EEG data with shape (n_channels, n_resampled_samples).
    """
    if eeg_data.ndim != 2:
        raise ValueError("Input EEG data must be 2-dimensional (n_channels, n_samples).")

    n_channels, n_samples = eeg_data.shape

    # Create an MNE Info object. Channel names are generic for now.
    ch_names = [f'EEG {i:03d}' for i in range(n_channels)]
    ch_types = ['eeg'] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Create a RawArray object from the numpy data
    raw = mne.io.RawArray(eeg_data, info, verbose=False)

    # 1. Resampling
    if sfreq != target_sfreq:
        raw.resample(sfreq=target_sfreq, verbose=False)

    # 2. Bandpass Filtering
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', filter_length='auto', verbose=False)

    # 3. Notch Filtering
    if notch_freqs is not None:
        raw.notch_filter(freqs=notch_freqs, method='iir', verbose=False)

    # Get the processed data as a numpy array
    processed_data = raw.get_data()

    # 4. Z-score Normalization (per channel)
    scaler = StandardScaler()
    processed_data_normalized = scaler.fit_transform(processed_data.T).T # Fit_transform expects (n_samples, n_features), so transpose twice

    return processed_data_normalized
