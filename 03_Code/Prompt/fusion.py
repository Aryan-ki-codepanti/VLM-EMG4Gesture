# ============================================================
# Prompt Module — fusion.py
# ============================================================

def normalize_text(value):
    """
    Normalize text for simple comparisons.
    """
    if value is None:
        return ""

    return str(value).strip().lower()


def neural_gesture_agreement(
    eeg_output,
    emg_output
):
    """
    Determine whether EEG and EMG predict the same
    semantic gesture.
    """

    eeg_semantic = eeg_output.get(
        "semantic_label"
    )

    emg_semantic = emg_output.get(
        "semantic_label"
    )

    if (
        eeg_semantic is None
        or emg_semantic is None
    ):
        return False

    return eeg_semantic == emg_semantic


def vision_has_object(vision_output):
    """
    Check whether Vision has identified a primary object.
    """
    return bool(
        vision_output.get("primary_object")
    )


def vision_has_active_interaction(
    vision_output
):
    """
    Check whether Vision reports active contact.
    """

    return (
        vision_output.get("contact_state")
        == 1
    )


def object_matches_instruction(
    instruction,
    vision_output
):
    """
    Very simple baseline object consistency check.

    This is deliberately conservative. It does not attempt
    sophisticated language understanding.
    """

    instruction = normalize_text(
        instruction
    )

    primary_object = normalize_text(
        vision_output.get(
            "primary_object"
        )
    )

    if not instruction or not primary_object:
        return False

    return primary_object in instruction


def deterministic_fusion(multimodal_context):
    """
    Produce a baseline deterministic decision from the
    normalized multimodal context.

    This is NOT the LLM.
    It is an interpretable fusion baseline.
    """

    task = multimodal_context["task"]
    eeg = multimodal_context["eeg"]
    emg = multimodal_context["emg"]
    vision = multimodal_context["vision"]

    instruction = task["instruction"]

    # --------------------------------------------------------
    # Neural agreement
    # --------------------------------------------------------

    neural_agreement = neural_gesture_agreement(
        eeg,
        emg
    )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    eeg_conf = eeg.get("confidence")
    emg_conf = emg.get("confidence")

    neural_confidences = [
        c for c in [eeg_conf, emg_conf]
        if c is not None
    ]

    average_neural_confidence = (
        sum(neural_confidences) /
        len(neural_confidences)
        if neural_confidences
        else 0.0
    )

    # --------------------------------------------------------
    # Vision evidence
    # --------------------------------------------------------

    object_present = vision_has_object(vision)

    active_interaction = vision_has_active_interaction(
        vision
    )

    object_matches = object_matches_instruction(
        instruction,
        vision
    )

    # --------------------------------------------------------
    # Baseline decision
    # --------------------------------------------------------

    if not object_present:
        decision = "WAIT"

    elif not object_matches:
        decision = "WAIT"

    elif not neural_agreement:
        decision = "WAIT"

    elif average_neural_confidence < 0.60:
        decision = "WAIT"

    elif not active_interaction:
        decision = "WAIT"

    else:
        decision = "VALID"

    return {
        "neural_gesture_agreement": neural_agreement,
        "average_neural_confidence": average_neural_confidence,
        "vision_object_present": object_present,
        "vision_active_interaction": active_interaction,
        "vision_object_matches_instruction": object_matches,
        "decision": decision,
    }
