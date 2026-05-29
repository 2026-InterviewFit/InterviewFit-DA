from enum import Enum

class ErrorCode(str, Enum):
    INVALID_FRAME = "INVALID_FRAME"
    FRAME_NOT_FOUND = "FRAME_NOT_FOUND"
