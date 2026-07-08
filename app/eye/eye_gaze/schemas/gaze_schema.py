from pydantic import BaseModel


class GazeAnalyzeRequest(BaseModel):
    video_path: str


class GazeAnalyzeResponse(BaseModel):
    total_frames: int
    analyzed_frames: int
    center_count: int
    left_count: int
    right_count: int
    no_face_count: int
    away_count: int
    gaze_ratio: float
    gaze_score: int