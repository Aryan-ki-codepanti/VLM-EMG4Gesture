
def normalize_hands23_contact(value):
    value = str(value).strip().lower()

    mapping = {
        "no_contact": "no_contact",
        "object_contact": "object_contact",
        "inconclusive": "inconclusive",
        "none": "unknown",
    }

    return mapping.get(value, "unknown")


def normalize_hands23_handedness(value):
    value = str(value).strip().lower()

    if value in {"left_hand", "left"}:
        return "left"

    if value in {"right_hand", "right"}:
        return "right"

    return "unknown"


def normalize_hands23_grasp(value):
    value = str(value).strip()

    if value == "Pow-Pris":
        return "power_grasp"

    return "unknown"
