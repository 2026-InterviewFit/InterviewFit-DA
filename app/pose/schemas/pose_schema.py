from pydantic import BaseModel

class PoseAnalysisRequest(BaseModel):
    video_url: str

class PoseAnalysisResponse(BaseModel):
    timestamp: int
    shoulder_tilt: float
    face_direction: float
    shoulder_status: str
    face_status: str
