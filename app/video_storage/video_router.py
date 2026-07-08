from fastapi import APIRouter
from pydantic import BaseModel

from app.frame.service.frame_service import analyze_frames
from app.video_storage.minio_downloader import download_video_from_minio

video_router = APIRouter(
    prefix="/video_storage",
    tags=["Video"]
)


class VideoRequest(BaseModel):
    videoUrl: str


@video_router.post("/save")
def save_video(request: VideoRequest):
    return process_video(
        request,
        "video downloaded and analyzed"
    )


@video_router.post("/analyze")
def analyze_video(request: VideoRequest):
    return process_video(
        request,
        "FastAPI 분석 완료"
    )


def process_video(
    request: VideoRequest,
    message: str
) -> dict:

    local_video_path = download_video_from_minio(
        request.videoUrl
    )

    result = analyze_frames(
        local_video_path
    )

    return create_response(
        message,
        request.videoUrl,
        local_video_path,
        result
    )


def create_response(
    message: str,
    video_url: str,
    local_path: str,
    result: dict
) -> dict:

    return {
        "status": "success",
        "message": message,
        "videoUrl": video_url,
        "localPath": local_path,
        "result": result
    }