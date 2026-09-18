# ============================================================
# Prompt Module — context.py
# ============================================================

def build_vision_context(
    hands=None,
    handedness=None,
    contact_state=None,
    grasp_type=None,
    primary_object=None,
    secondary_object=None,
    hand_object_relation=None,
):
    """
    Build normalized vision context.

    Preferred interface:
        hands=[{...}, {...}]

    Legacy single-hand arguments are retained for compatibility.
    """

    if hands is not None:
        return {
            "hands": hands
        }

    # Backward-compatible single-hand representation
    return {
        "hands": [
            {
                "handedness": handedness,
                "contact_state": contact_state,
                "grasp_type": grasp_type,
                "primary_object": primary_object,
                "secondary_object": secondary_object,
                "hand_object_relation": hand_object_relation,
            }
        ]
    }


def build_multimodal_context(
    task_instruction,
    eeg_output,
    emg_output,
    vision_output,
):
    """
    Combine normalized EEG, EMG, Vision and Text information
    into one standardized multimodal context.

    This object will later become the input to the Prompt Builder.
    """

    if not isinstance(eeg_output, dict):
        raise TypeError("eeg_output must be a dictionary.")

    if not isinstance(emg_output, dict):
        raise TypeError("emg_output must be a dictionary.")

    if not isinstance(vision_output, dict):
        raise TypeError("vision_output must be a dictionary.")

    return {
        "task": {
            "instruction": task_instruction.strip()
        },
        "eeg": eeg_output,
        "emg": emg_output,
        "vision": vision_output,
    }
