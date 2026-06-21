import logging

from fastapi import APIRouter
from pydantic import BaseModel

from app.frame.service.frame_service import analyze_frames
from app.video_storage.minio_downloader import download_video_from_minio

logger = logging.getLogger(__name__)

video_router = APIRouter(
    prefix="/apis/videos",
    tags=["Video"]
)


class VideoRequest(BaseModel):
    videoUrl: str


def process_video(video_url: str):
    logger.info("Spring Boot에서 받은 videoUrl: %s", video_url)

    local_video_path = download_video_from_minio(video_url)

    result = analyze_frames(local_video_path)

    return local_video_path, result


@video_router.post("/save")
def save_video(request: VideoRequest):

    local_video_path, result = process_video(request.videoUrl)

    return {
        "status": "success",
        "message": "video downloaded and analyzed",
        "videoUrl": request.videoUrl,
        "localPath": local_video_path,
        "result": result
    }


@video_router.post("/analyze")
def analyze_video(request: VideoRequest):

    local_video_path, result = process_video(request.videoUrl)

    return {
        "status": "success",
        "message": "FastAPI 분석 완료",
        "videoUrl": request.videoUrl,
        "localPath": local_video_path,
        "result": result
    }