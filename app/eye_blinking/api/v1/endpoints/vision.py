import os
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.VideoStorage.minio_downloader import resolve_video_path
from app.eye_blinking.schemas.eye import (
    BlinkAnalysisResponse,
    EyeBlinkingAnalyzeRequest,
    EyeLandmarksResponse,
)
from app.eye_blinking.services.eye_analysis import (
    analyze_blinks_from_video,
    extract_eye_landmarks_from_image,
)
from app.eye_blinking.utils.file_validation import (
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
    validate_upload_file,
)


router = APIRouter(prefix="/vision", tags=["vision"])


def calculate_blink_score(blinks_per_minute: float) -> float:
    """분당 눈 깜빡임 횟수를 0~100점으로 변환한다."""

    if 8 <= blinks_per_minute <= 30:
        return 100.0

    if blinks_per_minute < 8:
        return round(max(0.0, (blinks_per_minute / 8) * 100), 2)

    return round(max(0.0, 100 - ((blinks_per_minute - 30) * 3)), 2)


def build_blink_response(result: dict) -> BlinkAnalysisResponse:
    """eye_analysis.py의 dict 결과를 Spring이 저장하기 쉬운 응답 DTO로 정리한다."""

    blinks_per_minute = float(result.get("blinks_per_minute", 0.0))
    evaluation = result.get("evaluation", result.get("feedback", ""))

    return BlinkAnalysisResponse(
        total_frames=result.get("total_frames", 0),
        fps=result.get("fps", 0.0),
        duration_seconds=result.get("duration_seconds", 0.0),
        face_detected_frames=result.get("face_detected_frames", 0),
        blink_count=result.get("blink_count", 0),
        blinks_per_minute=blinks_per_minute,
        average_ear=result.get("average_ear", 0.0),
        min_ear=result.get("min_ear"),
        max_ear=result.get("max_ear"),
        ear_threshold=result.get("ear_threshold"),
        ear_consec_frames=result.get("ear_consec_frames"),
        closed_frame_count=result.get("closed_frame_count"),
        blink_timestamps=result.get("blink_timestamps", []),
        evaluation=evaluation,
        blink_rate=blinks_per_minute,
        score=float(result.get("score", calculate_blink_score(blinks_per_minute))),
        feedback=result.get("feedback", evaluation),
    )


@router.post("/eye-landmarks", response_model=EyeLandmarksResponse)
async def analyze_eye_landmarks(image: UploadFile = File(...)):
    """이미지 1장에서 눈 랜드마크와 EAR 값을 추출하는 단독 테스트용 API."""

    validate_upload_file(
        upload_file=image,
        allowed_extensions=ALLOWED_IMAGE_EXTENSIONS,
        expected_type_prefix="image/",
        type_name="이미지",
    )

    image_bytes = await image.read()

    try:
        result = extract_eye_landmarks_from_image(image_bytes)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"분석 중 오류가 발생했습니다: {str(e)}",
        )

    return result


@router.post("/blinks", response_model=BlinkAnalysisResponse)
def analyze_blinks_from_spring(request: EyeBlinkingAnalyzeRequest):
    """
    Spring Boot에서 전달한 videoUrl을 받아 눈 깜빡임을 분석한다.

    팀원 VideoStorage 흐름과 맞추기 위해 videoUrl은 두 형태를 모두 지원한다.

    1. MinIO objectKey
       - 실제 통합 흐름
       - FE → Spring /videos/upload → MinIO 저장 → videoUrl='videos/...' 반환
       - Spring → FastAPI /api/v1/vision/blinks로 같은 videoUrl 전달
       - FastAPI가 resolve_video_path()에서 MinIO 파일을 임시 다운로드

    2. 로컬 파일 경로
       - Swagger 단독 테스트나 개발 중 임시 테스트용

    주의: VideoStorage의 frame_extractor는 30프레임마다 1장 추출하므로
    눈 깜빡임처럼 짧은 이벤트를 세기에는 부적합하다.
    따라서 eye-blink는 프레임 이미지 목록이 아니라 원본 영상을 다시 연속 프레임으로 읽는다.
    """

    try:
        video_path = resolve_video_path(request.video_url)
        result = analyze_blinks_from_video(video_path)
        return build_blink_response(result)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"분석 중 오류가 발생했습니다: {str(e)}",
        )


@router.post("/blinks/upload", response_model=BlinkAnalysisResponse)
async def analyze_blinks_from_upload(video: UploadFile = File(...)):
    """
    Swagger나 단독 테스트용 업로드 방식 API.

    최종 FE/Spring 통합에서는 /videos/upload로 MinIO에 저장한 뒤
    /api/eye-blinking/analyze에 videoUrl을 넘기는 방식을 사용한다.
    이 API는 eye-blink 기능만 빠르게 검증할 때 남겨둔다.
    """

    validate_upload_file(
        upload_file=video,
        allowed_extensions=ALLOWED_VIDEO_EXTENSIONS,
        expected_type_prefix="video/",
        type_name="비디오",
    )

    suffix = os.path.splitext(video.filename)[1] if video.filename else ".mp4"
    tmp_file_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            content = await video.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        result = analyze_blinks_from_video(tmp_file_path)
        return build_blink_response(result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"분석 중 오류가 발생했습니다: {str(e)}",
        )

    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)