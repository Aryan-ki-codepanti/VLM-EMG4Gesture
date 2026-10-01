
from .vision_schema import (
    VALID_HANDEDNESS,
    VALID_CONTACT_STATES,
    VALID_RELATIONS,
)


def normalize_value(value):
    if value is None:
        return "unknown"

    value = str(value).strip()

    if not value:
        return "unknown"

    return value


def parse_vision_output(raw_output):
    """
    Validate and normalize Gemma/Gemini vision output.
    """

    if not isinstance(raw_output, dict):
        raise ValueError(
            "Vision output must be a dictionary."
        )

    hands = raw_output.get("hands", [])

    if not isinstance(hands, list):
        raise ValueError(
            "Vision 'hands' field must be a list."
        )

    normalized_hands = []

    for hand in hands:

        if not isinstance(hand, dict):
            continue

        handedness = normalize_value(
            hand.get("handedness")
        ).lower()

        contact_state = normalize_value(
            hand.get("contact_state")
        ).lower()

        relation = normalize_value(
            hand.get("hand_object_relation")
        ).lower()

        primary_object = normalize_value(
            hand.get("primary_object")
        )

        grasp_type = normalize_value(
            hand.get("grasp_type")
        )

        if handedness not in VALID_HANDEDNESS:
            handedness = "unknown"

        if contact_state not in VALID_CONTACT_STATES:
            contact_state = "unknown"

        if relation not in VALID_RELATIONS:
            relation = "unknown"

        normalized_hands.append({
            "handedness": handedness,
            "contact_state": contact_state,
            "primary_object": primary_object,
            "hand_object_relation": relation,
            "grasp_type": grasp_type,
        })

    return {
        "hands": normalized_hands
    }
