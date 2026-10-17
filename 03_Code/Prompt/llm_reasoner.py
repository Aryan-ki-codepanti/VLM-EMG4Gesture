# ============================================================
# Prompt Module — llm_reasoner.py
# ============================================================

from google import genai


DEFAULT_MODEL = "gemini-3.1-flash-lite"


REASONING_SCHEMA = {
    "type": "OBJECT",

    "properties": {
        "inferred_intention": {
            "type": "STRING",
            "description": "The likely intended action inferred from the available multimodal evidence.",
        },

        "requested_action": {
            "type": "STRING",
            "description": "The action requested by the user instruction.",
        },

        "eeg_consistent": {
            "type": "BOOLEAN",
            "description": (
                "True if EEG meaningfully supports the requested action within "
                "its capabilities; False if support is not established. False "
                "does not necessarily mean contradictory evidence."
            ),
        },

        "emg_consistent": {
            "type": "BOOLEAN",
            "description": (
                "True if the EMG gesture prediction supports the requested action; "
                "False if support is not established. A compatible gesture does "
                "not prove object contact or action completion."
            ),
        },

        "vision_consistent": {
            "type": "BOOLEAN",
            "description": (
                "True if the supplied visual observations support the requested "
                "action; False if support is not established. Unknown or missing "
                "evidence does not necessarily contradict the request."
            ),
        },

        "overall_decision": {
            "type": "STRING",
            "enum": [
                "VALID",
                "WAIT",
                "REJECT",
            ],
            "description": "Overall decision based on the multimodal evidence.",
        },

        "confidence": {
            "type": "NUMBER",
            "description": "Confidence in the reasoning result, from 0 to 1.",
        },

        "reason": {
            "type": "STRING",
            "description": "Concise evidence-based explanation.",
        },
    },

    "required": [
        "inferred_intention",
        "requested_action",
        "eeg_consistent",
        "emg_consistent",
        "vision_consistent",
        "overall_decision",
        "confidence",
        "reason",
    ],
}


def create_gemini_client(api_key):

    if not api_key:
        raise ValueError(
            "A Gemini API key is required."
        )

    return genai.Client(
        api_key=api_key
    )


def run_reasoning(
    client,
    prompt,
    model=DEFAULT_MODEL,
):

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": REASONING_SCHEMA,
        }
    )

    return response
