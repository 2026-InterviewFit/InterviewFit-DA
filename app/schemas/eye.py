from typing import List
from pydantic import BaseModel


class EyeLandmark(BaseModel):
    index: int
    x: float
    y: float
    z: float
    pixel_x: int
    pixel_y: int


class EyeLandmarksResponse(BaseModel):
    face_detected: bool
    image_width: int
    image_height: int
    left_eye_landmarks: List[EyeLandmark]
    right_eye_landmarks: List[EyeLandmark]


class BlinkAnalysisResponse(BaseModel):
    total_frames: int
    fps: float
    duration_seconds: float
    face_detected_frames: int
    blink_count: int
    blinks_per_minute: float
    average_ear: float
    blink_timestamps: List[float]
    evaluation: str