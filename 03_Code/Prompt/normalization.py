# ============================================================
# Prompt Module — normalization.py
# ============================================================

from .mappings import EEG_GESTURE_MAP, EMG_LABEL_MAP


def normalize_eeg_output(raw_label, confidence=None):
    """
    Convert an EEG decoder/dataset gesture label into
    the canonical semantic representation.

    Expected raw EEG labels:
        G1, G2, ..., G7
    """

    raw_label = str(raw_label).strip().upper()

    if raw_label not in EEG_GESTURE_MAP:
        raise ValueError(
            f"Unknown EEG label: {raw_label}. "
            f"Expected one of {list(EEG_GESTURE_MAP.keys())}."
        )

    return {
        "raw_label": raw_label,
        "semantic_label": EEG_GESTURE_MAP[raw_label],
        "confidence": (
            None
            if confidence is None
            else float(confidence)
        ),
    }


def normalize_emg_output(raw_label, confidence=None):
    """
    Convert an EMG decoder class index into the canonical
    semantic representation.

    Expected raw EMG classes:
        0, 1, 2, 3
    """

    raw_label = int(raw_label)

    if raw_label not in EMG_LABEL_MAP:
        raise ValueError(
            f"Unknown EMG label: {raw_label}. "
            f"Expected one of {list(EMG_LABEL_MAP.keys())}."
        )

    return {
        "raw_label": raw_label,
        "semantic_label": EMG_LABEL_MAP[raw_label],
        "confidence": (
            None
            if confidence is None
            else float(confidence)
        ),
    }
