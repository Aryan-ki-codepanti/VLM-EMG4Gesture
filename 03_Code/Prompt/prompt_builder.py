# ============================================================
# Prompt Module — prompt_builder.py
# ============================================================

from .prompt_config import (
    SYSTEM_ROLE,
    OUTPUT_INSTRUCTIONS,
    PROMPT_VERSION,
)


def _format_vision_observation(vision):
    """
    Format vision output for the reasoning prompt.

    Supports multi-hand vision output using the 'hands' list.
    """

    hands = vision.get("hands", [])

    if not hands:
        return "No hand observations available."

    formatted_hands = []

    for index, hand in enumerate(hands, start=1):
        formatted_hands.append(
            f"""
Hand {index}:

- Handedness: {hand.get("handedness", "unknown")}
- Contact state: {hand.get("contact_state", "unknown")}
- Grasp type: {hand.get("grasp_type", "unknown")}
- Primary object: {hand.get("primary_object", "unknown")}
- Secondary object: {hand.get("secondary_object", "unknown")}
- Hand-object relation: {hand.get("hand_object_relation", "unknown")}
""".strip()
        )

    return "\n\n".join(formatted_hands)


def build_reasoning_prompt(multimodal_context):

    task = multimodal_context["task"]
    eeg = multimodal_context["eeg"]
    emg = multimodal_context["emg"]
    vision = multimodal_context["vision"]

    vision_observation = _format_vision_observation(vision)

    prompt = f"""
{SYSTEM_ROLE}

PROMPT VERSION:
{PROMPT_VERSION}


========================
USER / TASK
========================

{task.get("instruction", "Not provided")}


========================
EEG OBSERVATION
========================

Raw label:
{eeg.get("raw_label", "unknown")}

Semantic gesture:
{eeg.get("semantic_label", "unknown")}

Confidence:
{eeg.get("confidence", "unknown")}


========================
EMG OBSERVATION
========================

Raw label:
{emg.get("raw_label", "unknown")}

Semantic gesture:
{emg.get("semantic_label", "unknown")}

Confidence:
{emg.get("confidence", "unknown")}


========================
VISION OBSERVATION
========================

{vision_observation}


========================
REASONING TASK
========================

Determine the likely intended action using the available
evidence.

Assess whether:

1. EEG supports the requested action.
2. EMG supports the requested action.
3. Vision supports the requested action.
4. The modalities are mutually consistent.

Do not assume that a specific EEG or EMG gesture directly
means grasp, release, or neutral unless the available evidence
supports that interpretation.

Do not invent objects, contact, grasp types, or intentions
that are not present in the observations.

{OUTPUT_INSTRUCTIONS}
"""

    return prompt.strip()
