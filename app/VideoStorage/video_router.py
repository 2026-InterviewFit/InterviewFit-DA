from fastapi import APIRouter
from pydantic import BaseModel

from app.frame.service.frame_service import analyze_frames
from app.VideoStorage.minio_downloader import download_video_from_minio

video_router = APIRouter(
    prefix="/VideoStorage",
    tags=["Video"]
)


class VideoRequest(BaseModel):
    videoUrl: str


@video_router.post("/save")
def save_video(request: VideoRequest):

    print("Spring Boot에서 받은 videoUrl:", request.videoUrl)

    local_video_path = download_video_from_minio(request.videoUrl)

    result = analyze_frames(local_video_path)

    return {
        "status": "success",
        "message": "video downloaded and analyzed",
        "videoUrl": request.videoUrl,
        "localPath": local_video_path,
        "result": result
    }


@video_router.post("/analyze")
def analyze_video(request: VideoRequest):

    print("Spring Boot에서 받은 videoUrl:", request.videoUrl)

    local_video_path = download_video_from_minio(request.videoUrl)

    result = analyze_frames(local_video_path)

    return {
        "status": "success",
        "message": "FastAPI 분석 완료",
        "videoUrl": request.videoUrl,
        "localPath": local_video_path,
        "result": result
    }