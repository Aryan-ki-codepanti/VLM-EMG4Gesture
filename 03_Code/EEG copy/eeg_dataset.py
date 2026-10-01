
import torch
from torch.utils.data import Dataset, DataLoader
import os
from eeg_data_preparation import load_eeg_data, DEFAULT_SFREQ # Use absolute import
from eeg_preprocessing import preprocess_eeg_data # Use absolute import

class EEGDataset(Dataset):
    """
    A custom PyTorch Dataset for EEG data.
    Loads data, applies preprocessing, and provides samples suitable for a DataLoader.
    Each item in the dataset corresponds to one complete EEG trial (one file).
    """
    def __init__(self, file_paths, sfreq=DEFAULT_SFREQ, target_sfreq=DEFAULT_SFREQ, l_freq=1, h_freq=30, notch_freqs=50):
        """
        Args:
            file_paths (list): A list of file paths to EEG .mat or .csv files.
            sfreq (float): Original sampling frequency of the EEG data.
            target_sfreq (float): Target sampling frequency for preprocessing.
            l_freq (float): Lower cutoff frequency for bandpass filter.
            h_freq (float): Higher cutoff frequency for bandpass filter.
            notch_freqs (float or list of floats): Frequencies to notch filter.
        """
        self.file_paths = file_paths
        self.sfreq = sfreq
        self.target_sfreq = target_sfreq
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.notch_freqs = notch_freqs

        self.data_cache = {}
        self.labels = []
        self.subjects = []
        self.runs = []
        self.original_gestures = []

        # Pre-process all data and store in cache
        print(f"Initializing EEGDataset with {len(self.file_paths)} files...")
        for i, filepath in enumerate(self.file_paths):
            # Suppress verbose output for MNE if processing many files
            # This can be controlled by `verbose=False` in MNE functions already handled in preprocess_eeg_data.
            if i % 100 == 0 and len(self.file_paths) > 100:
                print(f"  Processing file {i+1}/{len(self.file_paths)}: {os.path.basename(filepath)}")

            raw_eeg, metadata = load_eeg_data(filepath)
            preprocessed_eeg = preprocess_eeg_data(
                raw_eeg,
                sfreq=self.sfreq,
                target_sfreq=self.target_sfreq,
                l_freq=self.l_freq,
                h_freq=self.h_freq,
                notch_freqs=self.notch_freqs
            )
            self.data_cache[filepath] = preprocessed_eeg

            # Extract label (gesture) and metadata
            # Gesture G1-G7 maps to class labels 0-6
            gesture = metadata['gesture']
            self.labels.append(gesture - 1) # Map G1 to 0, G2 to 1, ..., G7 to 6
            self.subjects.append(metadata['subject'])
            self.runs.append(metadata['run'])
            self.original_gestures.append(gesture)
        print("Dataset initialization complete.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        filepath = self.file_paths[idx]
        eeg_data = self.data_cache[filepath]
        label = self.labels[idx]
        subject = self.subjects[idx]
        run = self.runs[idx]
        original_gesture = self.original_gestures[idx]

        # Convert to PyTorch tensors
        eeg_tensor = torch.from_numpy(eeg_data).float() # (n_channels, n_samples)
        label_tensor = torch.tensor(label).long() # Scalar

        metadata = {
            'subject': subject,
            'run': run,
            'original_gesture': original_gesture,
            'filepath': filepath
        }

        return eeg_tensor, label_tensor, metadata
