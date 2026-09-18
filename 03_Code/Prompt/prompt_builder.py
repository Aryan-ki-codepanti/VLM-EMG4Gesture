# ============================================================
# Prompt Module — prompt_builder.py
# ============================================================

from .prompt_config import (
    SYSTEM_ROLE,
    OUTPUT_INSTRUCTIONS,
    PROMPT_VERSION,
)


def build_reasoning_prompt(multimodal_context):

    task = multimodal_context["task"]
    eeg = multimodal_context["eeg"]
    emg = multimodal_context["emg"]
    vision = multimodal_context["vision"]

    prompt = f"""
{SYSTEM_ROLE}

PROMPT VERSION:
{PROMPT_VERSION}


========================
USER / TASK
========================

{task["instruction"]}


========================
EEG OBSERVATION
========================

Raw label:
{eeg.get("raw_label")}

Semantic gesture:
{eeg.get("semantic_label")}

Confidence:
{eeg.get("confidence")}


========================
EMG OBSERVATION
========================

Raw label:
{emg.get("raw_label")}

Semantic gesture:
{emg.get("semantic_label")}

Confidence:
{emg.get("confidence")}


========================
VISION OBSERVATION
========================

Handedness:
{vision.get("handedness")}

Contact state:
{vision.get("contact_state")}

Grasp type:
{vision.get("grasp_type")}

Primary object:
{vision.get("primary_object")}

Secondary object:
{vision.get("secondary_object")}

Hand-object relation:
{vision.get("hand_object_relation")}


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

{OUTPUT_INSTRUCTIONS}
"""

    return prompt.strip()
