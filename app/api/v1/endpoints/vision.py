import os
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.eye import BlinkAnalysisResponse, EyeLandmarksResponse
from app.services.eye_analysis import (
    analyze_blinks_from_video,
    extract_eye_landmarks_from_image,
)
from app.utils.file_validation import (
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
    validate_upload_file,
)

router = APIRouter(prefix="/vision", tags=["vision"])


@router.post("/eye-landmarks", response_model=EyeLandmarksResponse)
async def analyze_eye_landmarks(image: UploadFile = File(...)):
    """
    단일 이미지에서 눈 랜드마크를 추출하는 엔드포인트
    """
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
async def analyze_blinks(video: UploadFile = File(...)):
    """
    비디오 파일을 받아 프레임 단위로 눈 깜빡임을 분석하는 엔드포인트
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
        # 비디오는 OpenCV가 파일 경로를 통해 읽어야 하므로 임시 파일에 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            content = await video.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        result = analyze_blinks_from_video(tmp_file_path)

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

    return result