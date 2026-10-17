
"""
PyTorch Dataset for the BCI Competition IV 2b EEG dataset.

Each sample is one cue-aligned EEG trial extracted from a GDF recording.
Labels:
    0 = left hand  (event 769)
    1 = right hand (event 770)

Expected input:
    A list of GDF recording paths.

This dataset uses the new GDF pipeline:
    eeg_data_preparation.py
    eeg_preprocessing.py

Important:
    Split recordings by subject BEFORE fitting any learned preprocessing
    or training a model. Do not randomly split epochs across subjects for
    the LOSO experiment.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from eeg_data_preparation import (
    DEFAULT_SFREQ,
    load_eeg_gdf,
    extract_trial_metadata,
)
from eeg_preprocessing import (
    preprocess_gdf_raw,
    extract_eeg_epochs,
    DEFAULT_L_FREQ,
    DEFAULT_H_FREQ,
    DEFAULT_NOTCH_FREQS,
    DEFAULT_EPOCH_TMIN,
    DEFAULT_EPOCH_TMAX,
)


class EEGDataset(Dataset):
    """
    Load GDF recordings and expose their labelled EEG epochs.

    Each item returns:
        eeg_tensor: FloatTensor, shape (3, n_samples)
        label_tensor: LongTensor scalar, 0=left hand, 1=right hand
        metadata: dictionary describing the trial

    Parameters
    ----------
    file_paths : list[str | Path]
        Paths to BCI IV 2b GDF recordings. Training recordings should
        normally be sessions 01T, 02T and 03T.
    l_freq, h_freq : float
        Bandpass filter cutoffs in Hz.
    notch_freqs : float | list[float] | None
        Line-noise frequencies to remove. Set to None to skip notch filtering.
    tmin, tmax : float
        Epoch interval relative to the cue, in seconds.
    exclude_rejected : bool
        Exclude trials associated with event 1023, according to the current
        trial-metadata implementation.
    reject_unlabelled : bool
        Exclude trials without a recognised left/right cue.
    """

    def __init__(
        self,
        file_paths,
        l_freq=DEFAULT_L_FREQ,
        h_freq=DEFAULT_H_FREQ,
        notch_freqs=DEFAULT_NOTCH_FREQS,
        tmin=DEFAULT_EPOCH_TMIN,
        tmax=DEFAULT_EPOCH_TMAX,
        exclude_rejected=True,
        reject_unlabelled=True,
    ):
        self.file_paths = [str(Path(p)) for p in file_paths]

        if not self.file_paths:
            raise ValueError("EEGDataset received an empty file_paths list.")

        self.l_freq = l_freq
        self.h_freq = h_freq
        self.notch_freqs = notch_freqs
        self.tmin = tmin
        self.tmax = tmax
        self.exclude_rejected = exclude_rejected
        self.reject_unlabelled = reject_unlabelled

        # Each element represents one EEG epoch, not one complete GDF file.
        self.epochs = []
        self.labels = []
        self.trial_metadata = []

        print(
            f"Building EEGDataset from {len(self.file_paths)} GDF file(s)...")

        for file_index, filepath in enumerate(self.file_paths, start=1):
            print(
                f"  [{file_index}/{len(self.file_paths)}] "
                f"Processing {Path(filepath).name}"
            )

            raw, recording_metadata = load_eeg_gdf(
                filepath,
                preload=True,
            )

            try:
                trials = extract_trial_metadata(
                    raw,
                    metadata=recording_metadata,
                )

                # Filter the continuous EEG before extracting individual epochs.
                processed_raw = preprocess_gdf_raw(
                    raw,
                    l_freq=self.l_freq,
                    h_freq=self.h_freq,
                    notch_freqs=self.notch_freqs,
                    # Keep the original sampling rate so cue_sample indices
                    # remain aligned with the processed recording.
                    target_sfreq=None,
                )

                X, y, epoch_metadata = extract_eeg_epochs(
                    processed_raw,
                    trials,
                    tmin=self.tmin,
                    tmax=self.tmax,
                    exclude_rejected=self.exclude_rejected,
                    reject_unlabelled=self.reject_unlabelled,
                )

                if len(y) == 0:
                    print(
                        f"    Warning: no usable labelled epochs in "
                        f"{Path(filepath).name}"
                    )
                    continue

                if X.ndim != 3 or X.shape[1] != 3:
                    raise ValueError(
                        f"Expected epochs shaped (trials, 3, samples); "
                        f"got {X.shape} in {filepath}"
                    )

                if len(X) != len(y) or len(y) != len(epoch_metadata):
                    raise ValueError(
                        f"Epoch/label/metadata count mismatch in {filepath}: "
                        f"{len(X)}, {len(y)}, {len(epoch_metadata)}"
                    )

                for epoch_index in range(len(y)):
                    epoch = np.asarray(X[epoch_index], dtype=np.float32)
                    label = int(y[epoch_index])

                    if label not in (0, 1):
                        raise ValueError(
                            f"Unexpected label {label} in {filepath}; "
                            "expected 0=left or 1=right."
                        )

                    # Store a copy so epochs do not depend on temporary arrays.
                    self.epochs.append(epoch.copy())
                    self.labels.append(label)

                    row = epoch_metadata.iloc[epoch_index].to_dict()
                    row["filepath"] = filepath
                    row["filename"] = Path(filepath).name
                    row["subject"] = recording_metadata.get("subject")
                    row["session"] = recording_metadata.get("session")
                    row["split"] = recording_metadata.get("split")
                    self.trial_metadata.append(row)

            finally:
                # Release the recording's resources after its epochs are cached.
                raw.close()

        if not self.epochs:
            raise ValueError(
                "No usable EEG epochs were loaded. Check the GDF paths, "
                "event annotations, and preprocessing/epoching settings."
            )

        self.labels = np.asarray(self.labels, dtype=np.int64)

        print(f"Dataset ready: {len(self.epochs)} labelled EEG epochs.")
        print(
            "Class counts: "
            f"left={int(np.sum(self.labels == 0))}, "
            f"right={int(np.sum(self.labels == 1))}"
        )

    def __len__(self):
        return len(self.epochs)

    def __getitem__(self, idx):
        eeg_tensor = torch.from_numpy(self.epochs[idx]).float()
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)

        # Return a copy to prevent callers from mutating cached metadata.
        metadata = dict(self.trial_metadata[idx])

        return eeg_tensor, label_tensor, metadata

    def get_labels(self):
        """Return a copy of the labels, useful for class-count checks."""
        return self.labels.copy()

    def get_metadata(self):
        """Return a DataFrame with one row per retained epoch."""
        import pandas as pd

        return pd.DataFrame(self.trial_metadata)
