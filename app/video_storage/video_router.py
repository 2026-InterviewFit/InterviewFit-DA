from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.frame.service.frame_service import analyze_frames
from app.video_storage.minio_downloader import (
    resolve_video_path,
)


video_router = APIRouter(
    prefix="/video_storage",
    tags=["Video"],
)


class VideoRequest(BaseModel):
    videoUrl: str


@video_router.post("/analyze")
def analyze_video(
    request: VideoRequest,
) -> dict:
    """
    Spring Boot에서 전달한 MinIO objectKey 또는
    로컬 영상 경로를 받아 영상 프레임 분석을 수행한다.
    """

    try:
        local_video_path = resolve_video_path(
            request.videoUrl
        )

        result = analyze_frames(
            local_video_path
        )

        return {
            "status": "success",
            "message": "FastAPI 영상 분석 완료",
            "videoUrl": request.videoUrl,
            "localPath": local_video_path,
            "result": result,
        }

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "영상 분석 중 오류가 발생했습니다: "
                f"{str(error)}"
            ),
        ) from error