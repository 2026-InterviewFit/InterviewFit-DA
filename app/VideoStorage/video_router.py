from fastapi import APIRouter
from pydantic import BaseModel
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

    result = analyze_frames(request.videoUrl)

    return {
        "message": "video saved",
        "result": result
    }