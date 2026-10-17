
# ============================================================
# Prompt Module — prompt_builder.py
# ============================================================

from .prompt_config import (
    SYSTEM_ROLE,
    OUTPUT_INSTRUCTIONS,
    PROMPT_VERSION,
)


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _format_vision_observation(vision):
    vision = _as_dict(vision)
    hands = vision.get("hands", []) or []
    objects = vision.get("objects", []) or []

    lines = []

    if objects:
        lines.append(f"Detected objects: {objects}")

    if not hands:
        lines.append("No hand observations available.")
    else:
        for index, hand in enumerate(hands, start=1):
            if not isinstance(hand, dict):
                continue

            lines.append(
                f"""Hand observation {index}:
- Handedness field: {hand.get("handedness", "unknown")}
- Contact state: {hand.get("contact_state", "unknown")}
- Grasp type: {hand.get("grasp_type", "unknown")}
- Primary object: {hand.get("primary_object", "unknown")}
- Secondary object: {hand.get("secondary_object", "unknown")}
- Hand-object relation: {hand.get("hand_object_relation", "unknown")}"""
            )

    return "\n\n".join(lines)


def build_reasoning_prompt(multimodal_context):
    context = _as_dict(multimodal_context)

    task = _as_dict(context.get("task"))
    eeg = _as_dict(context.get("eeg"))
    emg = _as_dict(context.get("emg"))
    vision = _as_dict(context.get("vision"))

    instruction = (
        task.get("instruction")
        or context.get("task_instruction")
        or "Not provided"
    )

    vision_observation = _format_vision_observation(vision)

    prompt = f"""
{SYSTEM_ROLE}

PROMPT VERSION: {PROMPT_VERSION}

========================
REQUESTED ACTION
========================
{instruction}

========================
EEG OBSERVATION
========================
Task: {eeg.get("task", "unknown")}
Predicted class: {eeg.get("predicted_class", "unknown")}
Class ID: {eeg.get("class_id", "unknown")}
Dataset label: {eeg.get("dataset_label", "unknown")}
Probabilities: {eeg.get("probabilities", "not supplied")}
Confidence: {eeg.get("confidence", "not supplied")}
Interpretation note: {eeg.get("interpretation_note", "not supplied")}

Treat this as left/right motor-imagery evidence, not a grasp label.

========================
EMG OBSERVATION
========================
Predicted gesture: {
        emg.get("semantic_label")
        or emg.get("predicted_class")
        or emg.get("gesture")
        or "unknown"
    }
Raw label: {emg.get("raw_label", "unknown")}
Confidence: {emg.get("confidence", "not supplied")}

Treat this as a trained gesture-class prediction, not proof of
object contact or successful action completion.

========================
VISION OBSERVATION
========================
{vision_observation}

========================
REASONING TASK
========================
Assess the requested action using only the observations above.

For each modality, determine whether it meaningfully supports the
request within that modality's capabilities. Do not conflate lack
of support with evidence of contradiction.

In particular:
- Do not map EEG left/right motor imagery directly to grasp type.
- Do not compare EEG class with visible handedness unless the
  experimental mapping and handedness field semantics are validated.
- Do not infer object contact from an EMG grasp prediction.
- Do not infer a completed grasp from object visibility alone.
- Treat unknown or missing observations as uncertainty.
- Use WAIT when evidence is insufficient for the requested action.
- Explain which evidence supports the request and what remains
  unverified.

{OUTPUT_INSTRUCTIONS}
"""

    return prompt.strip()
