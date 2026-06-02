from fastapi import APIRouter
from pydantic import BaseModel

from app.frame.service.frame_service import analyze_frames

video_router = APIRouter(
    prefix="/VideoStorage",
    tags=["Video"]
)


class VideoRequest(BaseModel):
    videoUrl: str


@video_router.post("/save")
def save_video(request: VideoRequest):

    print("Spring Boot에서 받은 videoUrl:", request.videoUrl)

    result = analyze_frames(request.videoUrl)

    return {
        "status": "success",
        "message": "video saved and analyzed",
        "videoUrl": request.videoUrl,
        "result": result
    }


@video_router.post("/analyze")
def analyze_video(request: VideoRequest):

    print("Spring Boot에서 받은 videoUrl:", request.videoUrl)

    result = analyze_frames(request.videoUrl)

    return {
        "status": "success",
        "message": "FastAPI 분석 완료",
        "videoUrl": request.videoUrl,
        "result": result
    }