from enum import Enum

class ErrorCode(str, Enum):
    IMAGE_READ_FAILED = "IMAGE_READ_FAILED"
    INVALID_FRAME = "INVALID_FRAME"