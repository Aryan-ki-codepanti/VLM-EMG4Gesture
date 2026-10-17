
"""
eeg_data_preparation.py

Data loading and trial metadata extraction for the EEG branch.

New pipeline:
    BCI Competition IV Dataset 2b (.gdf)
    - 9 subjects
    - 5 sessions per subject
    - Left-hand motor imagery: event 769
    - Right-hand motor imagery: event 770
    - Trial start: event 768
    - Rejected-trial marker: event 1023

Legacy compatibility:
    The previous .mat/.csv loading functions are retained, but are
    not used by the new BCI 2b training pipeline.
"""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# Expected sampling frequency for BCI Competition IV Dataset 2b.
# Actual GDF sampling frequency is always read from the recording.
DEFAULT_SFREQ = 250

# Sampling frequency assumption retained only for legacy .mat/.csv data.
LEGACY_DEFAULT_SFREQ = 200

EEG_CHANNELS = ["EEG:C3", "EEG:Cz", "EEG:C4"]

TRIAL_START_EVENT = "768"

LEFT_CUE_EVENT = "769"
RIGHT_CUE_EVENT = "770"

REJECTED_TRIAL_EVENT = "1023"

CLASS_LABELS = {
    0: "left",
    1: "right",
}

EVENT_TO_LABEL = {
    LEFT_CUE_EVENT: "left",
    RIGHT_CUE_EVENT: "right",
}

LABEL_TO_ID = {
    "left": 0,
    "right": 1,
}

# Official BCI 2b filename convention, e.g. B0101T.gdf:
# B + two-digit subject + two-digit session + T/E.
GDF_FILENAME_PATTERN = re.compile(
    r"^B(?P<subject>\d{2})(?P<session>\d{2})(?P<split>[TE])\.gdf$",
    re.IGNORECASE,
)


# ---------------------------------------------------------
# Filename and directory utilities
# ---------------------------------------------------------

def parse_gdf_filename(filepath):
    """
    Extract subject, session, and recording split from a BCI 2b filename.

    Example:
        B0101T.gdf -> subject=1, session=1, split='T'

    Returns
    -------
    dict
        Metadata parsed from the filename.

    Raises
    ------
    ValueError
        If the filename does not match the expected BCI 2b convention.
    """
    filename = os.path.basename(str(filepath))
    match = GDF_FILENAME_PATTERN.match(filename)

    if match is None:
        raise ValueError(
            f"Filename does not match the expected BCI 2b format: "
            f"{filename}. Expected a name such as B0101T.gdf."
        )

    subject = int(match.group("subject"))
    session = int(match.group("session"))
    split = match.group("split").upper()

    if not 1 <= subject <= 9:
        raise ValueError(
            f"Unexpected subject number {subject} in {filename}."
        )

    if not 1 <= session <= 5:
        raise ValueError(
            f"Unexpected session number {session} in {filename}."
        )

    expected_split = "T" if session <= 3 else "E"

    if split != expected_split:
        raise ValueError(
            f"Unexpected split '{split}' for session {session} "
            f"in {filename}; expected '{expected_split}'."
        )

    return {
        "filename": filename,
        "subject": subject,
        "session": session,
        "split": split,
        "is_training_session": split == "T",
        "is_evaluation_session": split == "E",
    }


def discover_gdf_files(data_dir, recursive=True):
    """
    Discover GDF files under a directory.

    Parameters
    ----------
    data_dir : str or pathlib.Path
        Directory containing the recordings.
    recursive : bool
        If True, search subdirectories as well.

    Returns
    -------
    list[pathlib.Path]
        Sorted list of GDF file paths.
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"EEG data directory does not exist: {data_dir}"
        )

    if not data_dir.is_dir():
        raise NotADirectoryError(
            f"Expected a directory, received: {data_dir}"
        )

    if recursive:
        paths = data_dir.rglob("*")
    else:
        paths = data_dir.glob("*")

    files = sorted(
        path for path in paths
        if path.is_file() and path.suffix.lower() == ".gdf"
    )

    return files


# ---------------------------------------------------------
# GDF loading
# ---------------------------------------------------------

def load_eeg_gdf(filepath, preload=False):
    """
    Load a BCI 2b GDF recording using MNE.

    The recording's actual sampling frequency and channel metadata
    are read from the file. No resampling or filtering is performed here.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to a .gdf recording.
    preload : bool
        Whether to load the signal data into memory.

    Returns
    -------
    raw : mne.io.BaseRaw
        MNE raw recording, including its annotations.
    metadata : dict
        Filename and recording metadata.
    """
    import mne

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"GDF file not found: {filepath}")

    if filepath.suffix.lower() != ".gdf":
        raise ValueError(
            f"Expected a .gdf file, received: {filepath.name}"
        )

    metadata = parse_gdf_filename(filepath)

    raw = mne.io.read_raw_gdf(
        str(filepath),
        preload=preload,
        verbose="ERROR",
    )

    metadata["filepath"] = str(filepath)
    metadata["sfreq"] = float(raw.info["sfreq"])
    metadata["n_channels"] = len(raw.ch_names)
    metadata["channel_names"] = list(raw.ch_names)
    metadata["n_samples"] = int(raw.n_times)
    metadata["duration_seconds"] = (
        raw.n_times / raw.info["sfreq"]
    )

    # Record expected EEG channel availability explicitly.
    metadata["has_required_eeg_channels"] = all(
        channel in raw.ch_names for channel in EEG_CHANNELS
    )

    return raw, metadata


def load_eeg_data(filepath):
    """
    Load a GDF recording and return its metadata.

    This is the generic entry point for the new BCI 2b pipeline.
    Legacy .mat/.csv files should use their dedicated functions below.
    """
    filepath = Path(filepath)

    if filepath.suffix.lower() == ".gdf":
        return load_eeg_gdf(filepath)

    raise ValueError(
        f"Unsupported format for the new BCI 2b pipeline: "
        f"{filepath.suffix}. Expected .gdf."
    )


# ---------------------------------------------------------
# Event extraction and trial metadata
# ---------------------------------------------------------

def extract_trial_metadata(raw, metadata=None):
    """
    Extract labelled motor-imagery trial metadata from an MNE recording.

    A trial begins at event 768. The first subsequent left/right cue
    before the next trial start determines its label:
        769 -> left, label_id=0
        770 -> right, label_id=1

    Event 1023 is associated with the trial interval in which it occurs.
    Such trials are marked as rejected; they are not silently deleted.

    This function extracts metadata only. It does not extract signal
    epochs, filter signals, or choose an epoch time window.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Recording with annotations.
    metadata : dict or None
        Optional metadata from load_eeg_gdf().

    Returns
    -------
    pandas.DataFrame
        One row per trial-start event, including unlabelled or rejected
        trials so that trial handling remains explicit.
    """
    import mne

    if metadata is None:
        metadata = {}

    sfreq = float(raw.info["sfreq"])

    events, event_id = mne.events_from_annotations(
        raw,
        event_id="auto",
        verbose="ERROR",
    )

    id_to_description = {
        int(event_code): str(description)
        for description, event_code in event_id.items()
    }

    # Convert MNE event IDs back to their original annotation descriptions.
    event_rows = []

    for sample, _, event_code in events:
        description = id_to_description.get(
            int(event_code), str(event_code)
        )

        event_rows.append({
            "sample": int(sample),
            "time_seconds": float(sample / sfreq),
            "event": description,
        })

    event_df = pd.DataFrame(
        event_rows,
        columns=["sample", "time_seconds", "event"],
    )

    if event_df.empty:
        return pd.DataFrame(columns=[
            "filename",
            "subject",
            "session",
            "split",
            "trial_index",
            "trial_start_sample",
            "trial_start_seconds",
            "cue_sample",
            "cue_seconds",
            "cue_event",
            "label",
            "label_id",
            "is_rejected",
            "rejection_samples",
            "sfreq",
        ])

    event_df = event_df.sort_values(
        ["sample", "event"],
        kind="stable",
    ).reset_index(drop=True)

    trial_start_indices = event_df.index[
        event_df["event"] == TRIAL_START_EVENT
    ].tolist()

    trial_records = []

    filename = metadata.get(
        "filename",
        getattr(raw, "filenames", [None])[0],
    )

    if filename is not None:
        filename = os.path.basename(str(filename))

    # Parse filename metadata if the caller did not provide it.
    parsed_metadata = {}
    if filename:
        try:
            parsed_metadata = parse_gdf_filename(filename)
        except ValueError:
            parsed_metadata = {}

    combined_metadata = {**parsed_metadata, **metadata}

    for trial_number, start_idx in enumerate(trial_start_indices, start=1):
        start_row = event_df.iloc[start_idx]
        start_sample = int(start_row["sample"])

        # The next trial start bounds this trial's event interval.
        if trial_number < len(trial_start_indices):
            next_start_idx = trial_start_indices[trial_number]
            next_start_sample = int(
                event_df.iloc[next_start_idx]["sample"]
            )
            trial_events = event_df[
                (event_df["sample"] >= start_sample)
                & (event_df["sample"] < next_start_sample)
            ]
        else:
            trial_events = event_df[
                event_df["sample"] >= start_sample
            ]

        # Use the first explicit left/right cue inside this trial.
        cue_events = trial_events[
            trial_events["event"].isin(EVENT_TO_LABEL.keys())
        ]

        cue_sample = None
        cue_seconds = None
        cue_event = None
        label = None
        label_id = None

        if not cue_events.empty:
            cue_row = cue_events.iloc[0]
            cue_sample = int(cue_row["sample"])
            cue_seconds = float(cue_row["time_seconds"])
            cue_event = str(cue_row["event"])
            label = EVENT_TO_LABEL[cue_event]
            label_id = LABEL_TO_ID[label]

        rejection_events = trial_events[
            trial_events["event"] == REJECTED_TRIAL_EVENT
        ]

        is_rejected = not rejection_events.empty

        record = {
            "filename": filename,
            "subject": combined_metadata.get("subject"),
            "session": combined_metadata.get("session"),
            "split": combined_metadata.get("split"),
            "trial_index": trial_number,
            "trial_start_sample": start_sample,
            "trial_start_seconds": float(start_sample / sfreq),
            "cue_sample": cue_sample,
            "cue_seconds": cue_seconds,
            "cue_event": cue_event,
            "label": label,
            "label_id": label_id,
            "is_rejected": bool(is_rejected),
            "rejection_samples": [
                int(sample)
                for sample in rejection_events["sample"].tolist()
            ],
            "sfreq": sfreq,
        }

        trial_records.append(record)

    return pd.DataFrame(trial_records)


def summarize_trial_metadata(trials):
    """
    Return compact counts for auditing extracted trial metadata.

    The summary includes labelled, unlabelled, and rejected trials.
    """
    if trials.empty:
        return {
            "n_trials": 0,
            "n_labelled": 0,
            "n_unlabelled": 0,
            "n_rejected": 0,
            "label_counts": {},
        }

    return {
        "n_trials": int(len(trials)),
        "n_labelled": int(trials["label"].notna().sum()),
        "n_unlabelled": int(trials["label"].isna().sum()),
        "n_rejected": int(trials["is_rejected"].sum()),
        "label_counts": {
            str(label): int(count)
            for label, count in
            trials["label"].value_counts(dropna=True).items()
        },
    }


# ---------------------------------------------------------
# Legacy .mat/.csv loading
# ---------------------------------------------------------

def _parse_legacy_filename(filepath):
    """
    Parse the previous S<subject>_R<run>_G<gesture> naming convention.
    """
    filename = os.path.basename(filepath)
    match = re.search(r"S(\d+)_R(\d+)_G(\d+)", filename)

    metadata = {}

    if match:
        metadata["subject"] = int(match.group(1))
        metadata["run"] = int(match.group(2))
        metadata["gesture"] = int(match.group(3))

    metadata["sfreq"] = LEGACY_DEFAULT_SFREQ

    return metadata


def load_eeg_mat(filepath):
    """
    Legacy loader for the previous .mat dataset.

    Assumes data is stored under the 'data' key and shaped
    (n_channels, n_samples).
    """
    mat_data = sio.loadmat(filepath)
    data = mat_data.get("data")

    if data is None:
        raise ValueError(
            f"'data' key not found in {filepath}. "
            f"Available keys: {list(mat_data.keys())}"
        )

    if not isinstance(data, np.ndarray):
        raise TypeError(
            f"Data loaded from {filepath} is not a NumPy array. "
            f"Type: {type(data)}"
        )

    metadata = _parse_legacy_filename(filepath)

    return data, metadata


def load_eeg_csv(filepath):
    """
    Legacy loader for the previous .csv dataset.

    Assumes the CSV values are already in
    (n_channels, n_samples) format.
    """
    df = pd.read_csv(filepath)
    data = df.values.astype(np.float64)

    metadata = _parse_legacy_filename(filepath)

    return data, metadata


def load_legacy_eeg_data(filepath):
    """
    Load a legacy .mat or .csv recording.
    """
    filepath = str(filepath).lower()

    if filepath.endswith(".mat"):
        return load_eeg_mat(filepath)

    if filepath.endswith(".csv"):
        return load_eeg_csv(filepath)

    raise ValueError(
        f"Unsupported legacy file format: {filepath}. "
        "Only .mat and .csv are supported."
    )
