import os
import tempfile
import traceback

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.video_storage.minio_downloader import (
    cleanup_temp_video,
    resolve_video_path,
)
from app.eye.eye_blinking.schemas.eye_schema import (
    BlinkAnalysisResponse,
    EyeBlinkingAnalyzeRequest,
    EyeLandmarksResponse,
)
from app.eye.eye_blinking.services.eye_service import (
    analyze_blinks_from_video,
    extract_eye_landmarks_from_image,
)
from app.eye.eye_blinking.utils.file_validation import (
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
    validate_upload_file,
)

router = APIRouter()


def calculate_blink_score(blinks_per_minute: float) -> float:
    """분당 눈 깜빡임 횟수를 0~100점으로 변환한다."""
    if 8 <= blinks_per_minute <= 30:
        return 100.0

    if blinks_per_minute < 8:
        return round(max(0.0, (blinks_per_minute / 8) * 100), 2)

    return round(max(0.0, 100 - ((blinks_per_minute - 30) * 3)), 2)


def build_blink_response(result: dict) -> BlinkAnalysisResponse:
    """eye_service.py의 dict 결과를 Spring이 저장하기 쉬운 응답 DTO로 정리한다."""
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


@router.post("/analyze", response_model=BlinkAnalysisResponse)
@router.post("/blinks", response_model=BlinkAnalysisResponse, include_in_schema=False)
def analyze_blinks_from_spring(request: EyeBlinkingAnalyzeRequest):
    """
    Spring Boot에서 전달한 videoUrl을 받아 눈 깜빡임을 분석한다.

    안정성을 위해 eye 분석은 MinIO 영상을 요청 전용 임시 파일로 내려받는다.
    분석이 끝난 뒤에는 해당 임시 파일만 안전하게 삭제한다.
    """

    video_path = None

    try:
        video_path = resolve_video_path(
            request.video_url,
            unique_temp=True,
        )

        result = analyze_blinks_from_video(video_path)
        return build_blink_response(result)

    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    except Exception as error:
        print("\n🚨 [눈 깜빡임 분석 치명적 오류 발생] 🚨")
        traceback.print_exc()
        print("=========================================\n")

        raise HTTPException(
            status_code=500,
            detail=f"분석 중 오류가 발생했습니다: {str(error)}",
        ) from error

    finally:
        if video_path:
            cleanup_temp_video(video_path)


@router.post(
    "/eye-landmarks",
    response_model=EyeLandmarksResponse,
    include_in_schema=False,
)
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
        return extract_eye_landmarks_from_image(image_bytes)

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"분석 중 오류가 발생했습니다: {str(error)}",
        ) from error


@router.post(
    "/analyze/upload",
    response_model=BlinkAnalysisResponse,
    include_in_schema=False,
)
async def analyze_blinks_from_upload(video: UploadFile = File(...)):
    """Swagger나 단독 테스트용 업로드 방식 API."""

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

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    except Exception as error:
        print("\n🚨 [눈 깜빡임 업로드 분석 오류] 🚨")
        traceback.print_exc()
        print("=========================================\n")

        raise HTTPException(
            status_code=500,
            detail=f"분석 중 오류가 발생했습니다: {str(error)}",
        ) from error

    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.remove(tmp_file_path)
            except OSError as error:
                print(
                    "[EyeBlinking 업로드 임시파일 삭제 실패] "
                    f"path={tmp_file_path}, error={error}"
                )
