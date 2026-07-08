from enum import Enum

class ShoulderStatus(str, Enum):
    GOOD = "GOOD"
    BAD = "BAD"
    NOT_DETECTED = "NOT_DETECTED"

class FaceStatus(str, Enum):
    GOOD = "GOOD"
    BAD = "BAD"
    NOT_DETECTED = "NOT_DETECTED"