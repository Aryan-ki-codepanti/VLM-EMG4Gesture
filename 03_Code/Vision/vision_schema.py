VISION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "hands": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "handedness": {
                        "type": "STRING"
                    },
                    "contact_state": {
                        "type": "STRING"
                    },
                    "primary_object": {
                        "type": "STRING"
                    },
                    "hand_object_relation": {
                        "type": "STRING"
                    },
                    "grasp_type": {
                        "type": "STRING"
                    }
                },
                "required": [
                    "handedness",
                    "contact_state",
                    "primary_object",
                    "hand_object_relation",
                    "grasp_type"
                ]
            }
        }
    },
    "required": ["hands"]
}

VALID_HANDEDNESS = {
    "left",
    "right",
    "unknown"
}

VALID_CONTACT_STATES = {
    "no_contact",
    "object_contact",
    "inconclusive",
    "unknown"
}

VALID_RELATIONS = {
    "held",
    "touched",
    "near",
    "unknown"
}
