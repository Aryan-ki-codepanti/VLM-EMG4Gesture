# ============================================================
# Prompt Module — schema.py
# ============================================================

LLM_OUTPUT_FIELDS = [
    "inferred_intention",
    "requested_action",
    "eeg_consistent",
    "emg_consistent",
    "vision_consistent",
    "overall_decision",
    "confidence",
    "reason",
]


VALID_DECISIONS = {
    "VALID",
    "WAIT",
    "REJECT",
}


def create_empty_reasoning_result():
    """
    Create the canonical structure expected from the
    LLM reasoning stage.
    """
    return {
        "inferred_intention": None,
        "requested_action": None,
        "eeg_consistent": None,
        "emg_consistent": None,
        "vision_consistent": None,
        "overall_decision": None,
        "confidence": None,
        "reason": None,
    }


def validate_reasoning_result(result):
    """
    Validate the structure returned by the reasoning model.

    This function does not decide whether the action is correct.
    It only checks whether the response has the expected format.
    """

    if not isinstance(result, dict):
        raise TypeError(
            "Reasoning result must be a dictionary."
        )

    missing_fields = [
        field
        for field in LLM_OUTPUT_FIELDS
        if field not in result
    ]

    if missing_fields:
        raise ValueError(
            f"Missing required fields: {missing_fields}"
        )

    boolean_fields = [
        "eeg_consistent",
        "emg_consistent",
        "vision_consistent",
    ]

    for field in boolean_fields:

        if not isinstance(result[field], bool):
            raise ValueError(
                f"{field} must be True or False."
            )

    decision = str(
        result["overall_decision"]
    ).upper()

    if decision not in VALID_DECISIONS:
        raise ValueError(
            f"Invalid overall_decision: {decision}. "
            f"Expected one of {VALID_DECISIONS}."
        )

    confidence = result["confidence"]

    if not isinstance(
        confidence,
        (int, float)
    ):
        raise ValueError(
            "confidence must be numeric."
        )

    if not 0.0 <= float(confidence) <= 1.0:
        raise ValueError(
            "confidence must be between 0 and 1."
        )

    for field in [
        "inferred_intention",
        "requested_action",
        "reason",
    ]:

        if not isinstance(
            result[field],
            str
        ):
            raise ValueError(
                f"{field} must be a string."
            )

    return True
