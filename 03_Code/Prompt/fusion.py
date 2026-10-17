
# ============================================================
# Prompt Module — fusion.py
# ============================================================

def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _confidence(output):
    value = _as_dict(output).get("confidence")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if 0.0 <= value <= 1.0 else None


def _first_text(*values):
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _observed_objects(vision):
    """Collect explicitly supplied object names from vision output."""
    vision = _as_dict(vision)
    objects = []

    raw_objects = vision.get("objects", []) or []
    if isinstance(raw_objects, list):
        for item in raw_objects:
            if isinstance(item, str) and item.strip():
                objects.append(item.strip().lower())
            elif isinstance(item, dict):
                name = _first_text(item.get("name"), item.get("label"))
                if name:
                    objects.append(name.lower())

    hands = vision.get("hands", []) or []
    if isinstance(hands, list):
        for hand in hands:
            if not isinstance(hand, dict):
                continue

            for key in (
                "primary_object",
                "secondary_object",
                "object",
                "object_name",
                "object_label",
            ):
                name = hand.get(key)
                if (
                    isinstance(name, str)
                    and name.strip()
                    and name.strip().lower() not in {
                        "unknown", "none", "null", "n/a"
                    }
                ):
                    objects.append(name.strip().lower())

    return sorted(set(objects))


def _vision_contact_states(vision):
    vision = _as_dict(vision)
    states = []

    hands = vision.get("hands", []) or []
    if isinstance(hands, list):
        for hand in hands:
            if isinstance(hand, dict):
                state = hand.get("contact_state")
                if isinstance(state, str) and state.strip():
                    states.append(state.strip().lower())

    return states


def deterministic_fusion(multimodal_context):
    """
    Conservative deterministic pre-check.

    This function does not establish high-level intention or replace
    the LLM. EEG and EMG label spaces are different and are not directly
    compared for agreement.
    """
    context = _as_dict(multimodal_context)

    task_value = context.get("task") or context.get("task_instruction")
    if isinstance(task_value, dict):
        task = str(task_value.get("instruction") or "").strip()
    else:
        task = str(task_value or "").strip()

    eeg = _as_dict(context.get("eeg"))
    emg = _as_dict(context.get("emg"))
    vision = _as_dict(context.get("vision"))

    eeg_label = _first_text(
        eeg.get("predicted_class"),
        eeg.get("semantic_label"),
        eeg.get("label"),
    )

    emg_label = _first_text(
        emg.get("semantic_label"),
        emg.get("predicted_class"),
        emg.get("gesture"),
        emg.get("label"),
    )

    eeg_present = eeg_label is not None
    emg_present = emg_label is not None
    vision_present = bool(vision)

    eeg_confidence = _confidence(eeg)
    emg_confidence = _confidence(emg)

    objects = _observed_objects(vision)
    contact_states = _vision_contact_states(vision)

    # These are observations, not a complete semantic verification.
    unknown_values = {"unknown", "none", "null", "n/a", ""}
    object_identified = bool(objects)

    contact_observed = any(
        state in {"contact", "in_contact", "touching", "grasping"}
        for state in contact_states
    )

    explicit_no_contact = any(
        state in {"no_contact", "not_in_contact", "no contact"}
        for state in contact_states
    )

    # This pre-check intentionally does not infer object mismatch
    # from keywords in the task instruction.
    object_mismatch = False

    eeg_note = (
        "EEG predicts a motor-imagery class only; its relationship "
        "to the requested object action has not been established."
        if eeg_present else
        "EEG prediction is missing."
    )

    emg_note = (
        f"EMG predicts '{emg_label}'. This supports a trained gesture "
        "class but does not prove object contact or action completion."
        if emg_present else
        "EMG prediction is missing."
    )

    if not task:
        decision = "WAIT"
        reason = "The task instruction is missing."
    elif not (eeg_present and emg_present and vision_present):
        decision = "WAIT"
        missing = []
        if not eeg_present:
            missing.append("EEG prediction")
        if not emg_present:
            missing.append("EMG prediction")
        if not vision_present:
            missing.append("vision output")
        reason = "Insufficient input: missing " + ", ".join(missing) + "."
    else:
        # This baseline does not decide VALID from classifier outputs
        # alone. The LLM and/or an explicit task-specific verifier must
        # assess whether the requested interaction is supported.
        decision = "WAIT"

        if explicit_no_contact and not contact_observed:
            reason = (
                "Vision reports no contact, so the requested interaction "
                "has not been visually established. Recheck with a suitable "
                "frame or additional evidence."
            )
        elif not object_identified:
            reason = (
                "The vision output does not identify a relevant object. "
                "The requested object interaction remains unverified."
            )
        elif not contact_observed:
            reason = (
                "A relevant object may be visible, but contact is not "
                "established by the supplied vision fields."
            )
        else:
            reason = (
                "All modality outputs are present and vision reports contact, "
                "but this baseline does not independently verify that the "
                "observed interaction matches the requested action. "
                "Use the LLM result and inspect the evidence."
            )

    return {
        "stage": "deterministic_precheck",
        "task_instruction": task or None,

        "eeg_task": eeg.get(
            "task", "motor_imagery_classification"
        ),
        "eeg_predicted_class": eeg_label,
        "eeg_confidence": eeg_confidence,
        "eeg_consistent": None,
        "eeg_note": eeg_note,

        "emg_predicted_class": emg_label,
        "emg_confidence": emg_confidence,
        "emg_consistent": None,
        "emg_note": emg_note,

        "neural_gesture_agreement": None,

        "observed_objects": objects,
        "object_identified": object_identified,
        "contact_states": contact_states,
        "contact_observed": contact_observed,
        "explicit_no_contact": explicit_no_contact,
        "object_mismatch_warning": object_mismatch,
        "vision_available": vision_present,

        "decision": decision,
        "reason": reason,
    }
