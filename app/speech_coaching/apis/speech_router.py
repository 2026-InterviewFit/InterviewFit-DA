import os
import tempfile
from pathlib import Path

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
)

from app.speech_coaching.services import speech_service

from app.speech_coaching.schemas.speech_schema import (
    SpeechAnalyzeRequest,
    SpeechTranscriptionResponse,
)


router = APIRouter()


# =========================================================
# 기존 전체 스피치 분석 API
# =========================================================

@router.post("/analyze")
async def analyze_interview(
    request: SpeechAnalyzeRequest,
):
    return speech_service.process_interview_analysis(
        source_type=request.source_type,
        source_value=request.source_value,
        job_category=request.job_category,
        interview_id=request.interview_id,
        interview_answer_id=request.interview_answer_id,
        question_id=request.question_id,
        original_question=request.original_question,
    )


# =========================================================
# 꼬리질문용 STT 전용 API
# =========================================================

@router.post(
    "/transcribe-file",
    response_model=SpeechTranscriptionResponse,
)
async def transcribe_answer_file(
    file: UploadFile = File(...),
):
    """
    기본 질문 또는 꼬리질문에 대한
    사용자 답변 음성을 Whisper로 STT한다.

    이 API에서는 다음 작업을 하지 않는다.

    - 습관어 분석
    - 반복 표현 분석
    - 말하기 속도 분석
    - 침묵 분석
    - Gemini 답변 평가

    꼬리질문 생성을 위해
    답변 텍스트만 빠르게 반환한다.

    FE MediaRecorder에서 전달되는
    audio/webm, video/webm 등의 파일을
    그대로 Whisper에 전달할 수 있다.
    """

    temp_path = None

    try:
        # -------------------------------------------------
        # 1. 업로드 파일 읽기
        # -------------------------------------------------

        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="업로드된 음성 파일이 비어 있습니다.",
            )


        # -------------------------------------------------
        # 2. 원본 확장자 유지
        # -------------------------------------------------

        original_filename = (
            file.filename
            or "answer.webm"
        )

        suffix = Path(
            original_filename
        ).suffix


        if not suffix:
            suffix = ".webm"


        # -------------------------------------------------
        # 3. 임시 파일 생성
        # -------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp_file:

            temp_file.write(
                contents
            )

            temp_path = (
                temp_file.name
            )


        # -------------------------------------------------
        # 4. 기존 Whisper 함수 재사용
        # -------------------------------------------------

        text = (
            speech_service
            .run_free_whisper_stt(
                temp_path
            )
        )


        # 기존 함수는 실패 시
        # "STT 변환 실패" 문자열을 반환하고 있으므로
        # 여기서 HTTP 오류로 변환한다.
        if (
            not text
            or not text.strip()
        ):
            raise HTTPException(
                status_code=422,
                detail=(
                    "음성에서 텍스트를 "
                    "인식하지 못했습니다."
                ),
            )


        if (
            text.strip()
            == "STT 변환 실패"
        ):
            raise HTTPException(
                status_code=500,
                detail="Whisper STT 변환에 실패했습니다.",
            )


        # -------------------------------------------------
        # 5. 텍스트만 반환
        # -------------------------------------------------

        return SpeechTranscriptionResponse(
            text=text.strip(),
        )


    except HTTPException:
        raise


    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "답변 음성 STT 처리 중 "
                f"오류가 발생했습니다: {error}"
            ),
        ) from error


    finally:
        # -------------------------------------------------
        # 임시 파일 제거
        # -------------------------------------------------

        if (
            temp_path
            and os.path.exists(
                temp_path
            )
        ):
            os.remove(
                temp_path
            )


        await file.close()