import numpy as np
import mne

path = "S015R04.edf"

raw = mne.io.read_raw_edf(
    path,
    preload=False,
    verbose=False
)

print("Sampling rate:", raw.info["sfreq"])
print("Number of channels:", len(raw.ch_names))
print("Annotations found:")

for onset, duration, description in zip(
    raw.annotations.onset,
    raw.annotations.duration,
    raw.annotations.description
):
    print(f"{onset:8.2f}s | {description}")


raw = mne.io.read_raw_edf(
    "S015R04.edf",
    preload=True,
    verbose=False
)

events, event_id = mne.events_from_annotations(raw)

# Verify the expected event codes exist
assert all(code in event_id for code in ["T1", "T2"])

# Extract only left/right imagery epochs
epochs = mne.Epochs(
    raw,
    events,
    event_id={
        "T1": event_id["T1"],
        "T2": event_id["T2"],
    },
    tmin=0.5,
    tmax=3.5,
    baseline=None,
    preload=True,
    verbose=False,
)

# Assign our own binary labels:
# T1 = left, T2 = right
y = np.where(
    epochs.events[:, 2] == event_id["T1"],
    0,
    1
)

X = epochs.get_data()

print("EEG input shape:", X.shape)
print("Label counts:", np.bincount(y))
print("Label mapping: 0 = left, 1 = right")
