import os
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile

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
    if 8 <= blinks_per_minute <= 30:
        return 100.0

    if blinks_per_minute < 8:
        return round(max(0.0, (blinks_per_minute / 8) * 100), 2)

    return round(max(0.0, 100 - ((blinks_per_minute - 30) * 3)), 2)


def build_blink_response(result: dict) -> BlinkAnalysisResponse:
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
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"분석 중 오류가 발생했습니다: {str(e)}",
        )

    return result


@router.post("/blinks", response_model=BlinkAnalysisResponse)
def analyze_blinks_from_spring(request: EyeBlinkingAnalyzeRequest):
    """
    Spring Boot에서 저장된 영상 경로(videoUrl)를 받아 눈 깜빡임을 분석한다.

    주의:
    Docker에서 사용할 경우 backend와 ai-server가 같은 volume 경로를 공유해야 한다.
    예:
    backend와 ai-server 모두 /shared/uploads 경로를 볼 수 있어야 함.
    """

    video_path = request.video_url

    if not os.path.exists(video_path):
        raise HTTPException(
            status_code=404,
            detail=f"영상 파일을 찾을 수 없습니다: {video_path}",
        )

    try:
        result = analyze_blinks_from_video(video_path)
        return build_blink_response(result)

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"분석 중 오류가 발생했습니다: {str(e)}",
        )


@router.post("/blinks/upload", response_model=BlinkAnalysisResponse)
async def analyze_blinks_from_upload(video: UploadFile = File(...)):
    """
    Swagger나 단독 테스트용 업로드 방식 API.
    Spring 연동은 /blinks JSON 방식 사용.
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
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"분석 중 오류가 발생했습니다: {str(e)}",
        )

    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)