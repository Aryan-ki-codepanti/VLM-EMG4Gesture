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
            "description": "Whether the EEG observation supports the requested action.",
        },

        "emg_consistent": {
            "type": "BOOLEAN",
            "description": "Whether the EMG observation supports the requested action.",
        },

        "vision_consistent": {
            "type": "BOOLEAN",
            "description": "Whether the vision observation supports the requested action.",
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
            "response_format": {
                "text": {
                    "mime_type": "application/json",
                    "schema": REASONING_SCHEMA,
                }
            }
        },
    )

    return response
