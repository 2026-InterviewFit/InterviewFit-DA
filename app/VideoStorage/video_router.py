from fastapi import APIRouter
from pydantic import BaseModel

video_router = APIRouter(prefix="/VideoStorage", tags=["Video"])

class VideoRequest(BaseModel):
    videoUrl: str

@video_router.post("/analyze")
def analyze_video(request: VideoRequest):
    return {
        "message": "영상 받음",
        "videoUrl": request.videoUrl
    }