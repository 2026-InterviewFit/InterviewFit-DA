from typing import List, Optional

from pydantic import BaseModel, Field


class GazeAnalyzeRequest(BaseModel):
    video_url: str
    user_id: Optional[int] = None
    interview_id: Optional[int] = None


class GazeTimestamp(BaseModel):
    timestamp: float
    direction: str


class GazeAnalysisResponse(BaseModel):
    total_frames: int
    fps: float
    duration_seconds: float

    face_detected_frames: int
    gaze_detected_frames: int

    center_frames: int
    left_frames: int
    right_frames: int
    up_frames: int
    down_frames: int

    center_ratio: float
    gaze_away_ratio: float
    gaze_away_count: int

    score: float
    evaluation: str

    gaze_away_timestamps: List[GazeTimestamp] = Field(
        default_factory=list
    )

    analyzed_frames: int
    no_face_frames: int
    closed_eye_frames: int
    face_detection_ratio: float

    average_ear: float
    average_horizontal_ratio: float
    average_vertical_ratio: float

    metadata_total_frames: int
    frame_interval: int
    ear_threshold: float
    gaze_away_consec_frames: int