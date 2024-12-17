"""MediaPipe Hands connections and MANO hand model enhancements."""

# Connections for the hand palm, thumb, and fingers
HAND_PALM_CONNECTIONS = ((0, 1), (0, 5), (0, 17), (5, 9), (9, 13), (13, 17))
HAND_THUMB_CONNECTIONS = ((1, 2), (2, 3), (3, 4))
HAND_INDEX_FINGER_CONNECTIONS = ((5, 6), (6, 7), (7, 8))
HAND_MIDDLE_FINGER_CONNECTIONS = ((9, 10), (10, 11), (11, 12))
HAND_RING_FINGER_CONNECTIONS = ((13, 14), (14, 15), (15, 16))
HAND_PINKY_FINGER_CONNECTIONS = ((17, 18), (18, 19), (19, 20))

# All hand bone connections combined
HAND_BONES = (
    HAND_PALM_CONNECTIONS
    + HAND_THUMB_CONNECTIONS
    + HAND_INDEX_FINGER_CONNECTIONS
    + HAND_MIDDLE_FINGER_CONNECTIONS
    + HAND_RING_FINGER_CONNECTIONS
    + HAND_PINKY_FINGER_CONNECTIONS
)

# Hand joint names as per the typical skeleton structure
HAND_JOINT_NAMES = (
    "WRIST",
    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",
    "INDEX_MCP",
    "INDEX_PIP",
    "INDEX_DIP",
    "INDEX_TIP",
    "MIDDLE_MCP",
    "MIDDLE_PIP",
    "MIDDLE_DIP",
    "MIDDLE_TIP",
    "RING_MCP",
    "RING_PIP",
    "RING_DIP",
    "RING_TIP",
    "PINKY_MCP",
    "PINKY_PIP",
    "PINKY_DIP",
    "PINKY_TIP",
)

# Parent-child relationships of hand joints (index refers to HAND_JOINT_NAMES)
# -1 indicates no parent (root node)
HAND_JOINT_PARENTS = [
    -1,  # WRIST
    0,  # THUMB_CMC
    1,  # THUMB_MCP
    2,  # THUMB_IP
    3,  # THUMB_TIP
    0,  # INDEX_MCP
    5,  # INDEX_PIP
    6,  # INDEX_DIP
    7,  # INDEX_TIP
    0,  # MIDDLE_MCP
    9,  # MIDDLE_PIP
    10,  # MIDDLE_DIP
    11,  # MIDDLE_TIP
    0,  # RING_MCP
    13,  # RING_PIP
    14,  # RING_DIP
    15,  # RING_TIP
    0,  # PINKY_MCP
    17,  # PINKY_PIP
    18,  # PINKY_DIP
    19,  # PINKY_TIP
]

# Additional faces added to the MANO hand mesh for watertightness
NEW_MANO_FACES = {
    "right": [
        [92, 38, 234],
        [234, 38, 239],
        [38, 122, 239],
        [239, 122, 279],
        [122, 118, 279],
        [279, 118, 215],
        [118, 117, 215],
        [215, 117, 214],
        [117, 119, 214],
        [214, 119, 121],
        [119, 120, 121],
        [121, 120, 78],
        [120, 108, 78],
        [78, 108, 79],
    ],
    "left": [
        [234, 38, 92],
        [239, 38, 234],
        [239, 122, 38],
        [279, 122, 239],
        [279, 118, 122],
        [215, 118, 279],
        [215, 117, 118],
        [214, 117, 215],
        [214, 119, 117],
        [121, 119, 214],
        [121, 120, 119],
        [78, 120, 121],
        [78, 108, 120],
        [79, 108, 78],
    ],
}

# Number of vertices and faces in the MANO model
NUM_MANO_VERTS = 778
NUM_MANO_FACES = 1538
