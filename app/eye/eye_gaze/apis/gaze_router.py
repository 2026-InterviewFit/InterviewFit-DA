from fastapi import APIRouter, HTTPException

from app.eye.eye_gaze.schemas.gaze_schema import (
    GazeAnalysisResponse,
    GazeAnalyzeRequest,
)
from app.eye.eye_gaze.services.gaze_service import (
    analyze_gaze_from_video,
)
from app.video_storage.minio_downloader import resolve_video_path


router = APIRouter()


@router.post(
    "/analyze",
    response_model=GazeAnalysisResponse,
)
def analyze_gaze(
    request: GazeAnalyzeRequest,
) -> GazeAnalysisResponse:
    """
    Spring Boot에서 전달한 영상 URL 또는 object key를 이용해
    시선 방향과 정면 응시율을 분석한다.
    """

    try:
        video_path = resolve_video_path(
            request.video_url
        )

        result = analyze_gaze_from_video(
            video_path
        )

        return GazeAnalysisResponse(
            **result
        )

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
                "시선 분석 중 오류가 발생했습니다: "
                f"{str(error)}"
            ),
        ) from error
