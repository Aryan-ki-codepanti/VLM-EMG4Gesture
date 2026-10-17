
"""
eeg_preprocessing.py

Preprocessing utilities for the EEG branch.

New pipeline:
    - Load continuous BCI 2b EEG from an MNE Raw object.
    - Select C3, Cz, and C4.
    - Filter continuous EEG before epoch extraction.
    - Extract cue-relative left/right motor-imagery epochs.
    - Exclude trials marked as rejected.
    - Preserve trial metadata for evaluation and auditing.

Legacy pipeline:
    preprocess_eeg_data() remains available for old .mat/.csv workflows.
"""

import numpy as np
import mne

from sklearn.preprocessing import StandardScaler

from eeg_data_preparation import (
    DEFAULT_SFREQ,
    EEG_CHANNELS,
    LEFT_CUE_EVENT,
    RIGHT_CUE_EVENT,
    LABEL_TO_ID,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# These are initial, configurable settings for experimentation.
# Confirm the final epoch window against the official protocol
# before reporting final model results.

DEFAULT_L_FREQ = 1.0
DEFAULT_H_FREQ = 40.0
DEFAULT_NOTCH_FREQS = 50.0

# Epoch window relative to the left/right cue onset, in seconds.
DEFAULT_EPOCH_TMIN = 0.5
DEFAULT_EPOCH_TMAX = 3.5

# No resampling is performed by default.
# BCI 2b recordings in the audited directory are already 250 Hz.


# ---------------------------------------------------------
# Legacy preprocessing
# ---------------------------------------------------------

def preprocess_eeg_data(
    eeg_data,
    sfreq=DEFAULT_SFREQ,
    target_sfreq=DEFAULT_SFREQ,
    l_freq=1.0,
    h_freq=30.0,
    notch_freqs=50.0,
):
    """
    Legacy preprocessing for an EEG array.

    Parameters
    ----------
    eeg_data : np.ndarray
        Array with shape (n_channels, n_samples).
    sfreq : float
        Original sampling frequency.
    target_sfreq : float
        Desired sampling frequency.
    l_freq : float
        Lower bandpass cutoff in Hz.
    h_freq : float
        Upper bandpass cutoff in Hz.
    notch_freqs : float, list, or None
        Frequencies to remove with a notch filter.

    Returns
    -------
    np.ndarray
        Preprocessed array with shape
        (n_channels, n_resampled_samples).

    Notes
    -----
    This retains the old whole-array normalisation behavior for
    compatibility. Do not use it to preprocess the entire dataset
    before a subject-wise train/test split.
    """
    eeg_data = np.asarray(eeg_data, dtype=np.float64)

    if eeg_data.ndim != 2:
        raise ValueError(
            "Input EEG data must be 2-dimensional "
            "(n_channels, n_samples)."
        )

    if eeg_data.shape[0] == 0 or eeg_data.shape[1] == 0:
        raise ValueError("Input EEG data cannot be empty.")

    if not np.isfinite(eeg_data).all():
        raise ValueError("Input EEG data contains NaN or infinite values.")

    if sfreq <= 0 or target_sfreq <= 0:
        raise ValueError("Sampling frequencies must be positive.")

    if l_freq is not None and h_freq is not None:
        if not 0 < l_freq < h_freq < sfreq / 2:
            raise ValueError(
                "Require 0 < l_freq < h_freq < Nyquist frequency."
            )

    ch_names = [
        f"EEG {i:03d}" for i in range(eeg_data.shape[0])
    ]

    info = mne.create_info(
        ch_names=ch_names,
        sfreq=float(sfreq),
        ch_types=["eeg"] * len(ch_names),
    )

    raw = mne.io.RawArray(
        eeg_data,
        info,
        verbose="ERROR",
    )

    if not np.isclose(sfreq, target_sfreq):
        raw.resample(
            sfreq=float(target_sfreq),
            verbose="ERROR",
        )

    raw.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        fir_design="firwin",
        filter_length="auto",
        verbose="ERROR",
    )

    if notch_freqs is not None:
        raw.notch_filter(
            freqs=notch_freqs,
            method="iir",
            verbose="ERROR",
        )

    processed_data = raw.get_data()

    # Legacy behavior retained for existing code.
    scaler = StandardScaler()
    processed_data_normalized = scaler.fit_transform(
        processed_data.T
    ).T

    return processed_data_normalized


# ---------------------------------------------------------
# Continuous GDF preprocessing
# ---------------------------------------------------------

def preprocess_gdf_raw(
    raw,
    l_freq=DEFAULT_L_FREQ,
    h_freq=DEFAULT_H_FREQ,
    notch_freqs=DEFAULT_NOTCH_FREQS,
    target_sfreq=None,
):
    """
    Prepare continuous EEG from a BCI 2b GDF recording.

    Selects C3, Cz, and C4, then filters the continuous signal.
    EOG channels are not passed to the classifier.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw GDF recording returned by load_eeg_gdf().
    l_freq : float
        High-pass cutoff in Hz.
    h_freq : float
        Low-pass cutoff in Hz.
    notch_freqs : float, list, or None
        Line-noise frequencies to remove.
    target_sfreq : float or None
        Optional target sampling frequency. None preserves the
        original sampling rate.

    Returns
    -------
    mne.io.BaseRaw
        Filtered, EEG-only copy of the recording.
    """
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("raw must be an MNE Raw object.")

    missing_channels = [
        channel for channel in EEG_CHANNELS
        if channel not in raw.ch_names
    ]

    if missing_channels:
        raise ValueError(
            f"Required EEG channels are missing: {missing_channels}. "
            f"Available channels: {raw.ch_names}"
        )

    processed_raw = raw.copy()
    processed_raw.pick(EEG_CHANNELS)

    sfreq = float(processed_raw.info["sfreq"])
    nyquist = sfreq / 2.0

    if l_freq is not None and l_freq <= 0:
        raise ValueError("l_freq must be positive or None.")

    if h_freq is not None and h_freq >= nyquist:
        raise ValueError(
            f"h_freq={h_freq} Hz must be below the Nyquist "
            f"frequency ({nyquist} Hz)."
        )

    if (
        l_freq is not None
        and h_freq is not None
        and l_freq >= h_freq
    ):
        raise ValueError("l_freq must be less than h_freq.")

    # Filter the continuous signal before extracting epochs.
    processed_raw.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        fir_design="firwin",
        filter_length="auto",
        verbose="ERROR",
    )

    if notch_freqs is not None:
        frequencies = np.atleast_1d(
            np.asarray(notch_freqs, dtype=float)
        )

        if np.any(frequencies <= 0) or np.any(frequencies >= nyquist):
            raise ValueError(
                "Notch frequencies must be positive and below Nyquist."
            )

        processed_raw.notch_filter(
            freqs=frequencies,
            method="iir",
            verbose="ERROR",
        )

    if target_sfreq is not None:
        if target_sfreq <= 0:
            raise ValueError("target_sfreq must be positive.")

        if not np.isclose(
            processed_raw.info["sfreq"], target_sfreq
        ):
            processed_raw.resample(
                sfreq=float(target_sfreq),
                verbose="ERROR",
            )

    return processed_raw


# ---------------------------------------------------------
# Cue-relative epoch extraction
# ---------------------------------------------------------

def extract_eeg_epochs(
    processed_raw,
    trials,
    tmin=DEFAULT_EPOCH_TMIN,
    tmax=DEFAULT_EPOCH_TMAX,
    exclude_rejected=True,
    reject_unlabelled=True,
):
    """
    Extract fixed-length EEG epochs using trial metadata.

    Epochs are anchored to the explicit left/right cue sample.
    Event 769 maps to left (0); event 770 maps to right (1).

    Parameters
    ----------
    processed_raw : mne.io.BaseRaw
        EEG-only recording returned by preprocess_gdf_raw().
    trials : pandas.DataFrame
        Trial metadata returned by extract_trial_metadata().
    tmin : float
        Start time relative to cue onset, in seconds.
    tmax : float
        End time relative to cue onset, in seconds.
    exclude_rejected : bool
        If True, exclude trials flagged is_rejected.
    reject_unlabelled : bool
        If True, skip trials without a left/right cue label.

    Returns
    -------
    X : np.ndarray
        EEG epochs, shape (n_trials, 3, n_times).
        Signal values remain in the units returned by MNE.
    y : np.ndarray
        Integer labels: 0=left, 1=right.
    epoch_metadata : pandas.DataFrame
        Metadata for each retained epoch, in the same order as X and y.

    Notes
    -----
    This function does not perform per-recording or per-subject
    StandardScaler fitting. Any learned normalization must be fitted
    on training data only within each cross-validation fold.
    """
    import pandas as pd

    if not isinstance(processed_raw, mne.io.BaseRaw):
        raise TypeError("processed_raw must be an MNE Raw object.")

    if not isinstance(trials, pd.DataFrame):
        raise TypeError("trials must be a pandas DataFrame.")

    if tmin < 0 or tmax <= tmin:
        raise ValueError(
            "Require 0 <= tmin < tmax for cue-relative epochs."
        )

    required_columns = {
        "cue_sample",
        "label",
        "label_id",
        "is_rejected",
    }

    missing_columns = required_columns.difference(trials.columns)

    if missing_columns:
        raise ValueError(
            f"Trial metadata is missing columns: "
            f"{sorted(missing_columns)}"
        )

    missing_channels = [
        channel for channel in EEG_CHANNELS
        if channel not in processed_raw.ch_names
    ]

    if missing_channels:
        raise ValueError(
            f"Processed recording is missing EEG channels: "
            f"{missing_channels}"
        )

    sfreq = float(processed_raw.info["sfreq"])
    raw_data = processed_raw.get_data(
        picks=EEG_CHANNELS
    )

    # Include both endpoints when calculating the epoch length.
    start_offset = int(round(tmin * sfreq))
    end_offset = int(round(tmax * sfreq))

    epoch_length = end_offset - start_offset + 1

    if epoch_length <= 1:
        raise ValueError("The requested epoch window is too short.")

    epochs = []
    labels = []
    metadata_records = []

    skipped_rejected = 0
    skipped_unlabelled = 0
    skipped_boundary = 0
    skipped_invalid_label = 0

    for _, trial in trials.iterrows():

        # Reject trials flagged by the data-preparation helper.
        if exclude_rejected and bool(trial["is_rejected"]):
            skipped_rejected += 1
            continue

        label = trial["label"]
        label_id = trial["label_id"]

        if pd.isna(label) or pd.isna(label_id):
            if reject_unlabelled:
                skipped_unlabelled += 1
                continue
            else:
                continue

        label = str(label).lower()
        label_id = int(label_id)

        # Validate that text labels and numeric labels agree.
        if (
            label not in LABEL_TO_ID
            or LABEL_TO_ID[label] != label_id
        ):
            skipped_invalid_label += 1
            continue

        cue_event = trial.get("cue_event", None)

        # Do not allow an inconsistent cue event to silently
        # override the class label.
        expected_label = {
            LEFT_CUE_EVENT: "left",
            RIGHT_CUE_EVENT: "right",
        }.get(str(cue_event))

        if expected_label is not None and expected_label != label:
            skipped_invalid_label += 1
            continue

        cue_sample = trial["cue_sample"]

        if pd.isna(cue_sample):
            skipped_unlabelled += 1
            continue

        cue_sample = int(cue_sample)

        start_sample = cue_sample + start_offset
        end_sample = cue_sample + end_offset

        # Do not pad or wrap trials at recording boundaries.
        if (
            start_sample < 0
            or end_sample >= raw_data.shape[1]
        ):
            skipped_boundary += 1
            continue

        epoch = raw_data[:, start_sample:end_sample + 1]

        if epoch.shape != (len(EEG_CHANNELS), epoch_length):
            skipped_boundary += 1
            continue

        if not np.isfinite(epoch).all():
            skipped_boundary += 1
            continue

        epochs.append(epoch.astype(np.float32))
        labels.append(label_id)

        record = trial.to_dict()
        record["epoch_start_sample"] = start_sample
        record["epoch_end_sample"] = end_sample
        record["epoch_tmin"] = float(tmin)
        record["epoch_tmax"] = float(tmax)
        record["epoch_n_samples"] = int(epoch.shape[1])

        metadata_records.append(record)

    if epochs:
        X = np.stack(epochs, axis=0)
    else:
        X = np.empty(
            (0, len(EEG_CHANNELS), epoch_length),
            dtype=np.float32,
        )

    y = np.asarray(labels, dtype=np.int64)
    epoch_metadata = pd.DataFrame(metadata_records)

    print("EEG epoch extraction summary")
    print(f"  Input trials:       {len(trials)}")
    print(f"  Extracted epochs:   {len(X)}")
    print(f"  Skipped rejected:   {skipped_rejected}")
    print(f"  Skipped unlabelled:  {skipped_unlabelled}")
    print(f"  Skipped boundaries:  {skipped_boundary}")
    print(f"  Skipped bad labels:  {skipped_invalid_label}")
    print(f"  Epoch shape:        {X.shape}")

    return X, y, epoch_metadata


# ---------------------------------------------------------
# End-to-end helper for one GDF recording
# ---------------------------------------------------------

def preprocess_gdf_trials(
    raw,
    trials,
    l_freq=DEFAULT_L_FREQ,
    h_freq=DEFAULT_H_FREQ,
    notch_freqs=DEFAULT_NOTCH_FREQS,
    target_sfreq=None,
    tmin=DEFAULT_EPOCH_TMIN,
    tmax=DEFAULT_EPOCH_TMAX,
    exclude_rejected=True,
):
    """
    Convenience function for preprocessing and epoching one GDF file.

    Returns
    -------
    X, y, epoch_metadata, processed_raw
    """
    processed_raw = preprocess_gdf_raw(
        raw=raw,
        l_freq=l_freq,
        h_freq=h_freq,
        notch_freqs=notch_freqs,
        target_sfreq=target_sfreq,
    )

    X, y, epoch_metadata = extract_eeg_epochs(
        processed_raw=processed_raw,
        trials=trials,
        tmin=tmin,
        tmax=tmax,
        exclude_rejected=exclude_rejected,
        reject_unlabelled=True,
    )

    return X, y, epoch_metadata, processed_raw
