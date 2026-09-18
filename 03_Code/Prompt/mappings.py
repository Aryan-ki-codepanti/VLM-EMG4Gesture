# ============================================================
# Prompt Module — mappings.py
# ============================================================

EEG_GESTURE_MAP = {
    "G1": "large_diameter_grasp",
    "G2": "medium_diameter_grasp",
    "G3": "three_finger_sphere_grasp",
    "G4": "prismatic_pinch_grasp",
    "G5": "power_grasp",
    "G6": "cut_grasp",
    "G7": "open_hand",
}


EMG_LABEL_MAP = {
    0: "large_diameter_grasp",
    1: "medium_diameter_grasp",
    2: "three_finger_sphere_grasp",
    3: "prismatic_pinch_grasp",
}


SEMANTIC_DISPLAY_NAMES = {
    "large_diameter_grasp": "Large Diameter Grasp",
    "medium_diameter_grasp": "Medium Diameter Grasp",
    "three_finger_sphere_grasp": "Three-Finger Sphere Grasp",
    "prismatic_pinch_grasp": "Prismatic Pinch Grasp",
    "power_grasp": "Power Grasp",
    "cut_grasp": "Cut Grasp",
    "open_hand": "Open Hand",
}
