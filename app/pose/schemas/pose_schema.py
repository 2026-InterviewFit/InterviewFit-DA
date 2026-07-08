from pydantic import BaseModel

class PoseAnalysisRequest(BaseModel):
    video_object_key: str

class PoseAnalysisResponse(BaseModel):
    timestamp: int
    shoulder_tilt: float | None = None
    face_direction: float | None = None
    shoulder_status: str
    face_status: str
