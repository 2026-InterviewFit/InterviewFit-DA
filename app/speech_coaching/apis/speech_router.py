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
    기본 질문 또는 꼬리질문 답변 음성을 STT한다.

    브라우저 MediaRecorder의 WebM/Opus 파일을 바로 Whisper에 넣기 전에
    ffmpeg 기반 기존 오디오 추출 함수를 사용해 16kHz mono WAV로 변환한다.
    이 과정으로 브라우저별 WebM 컨테이너 차이의 영향을 줄인다.
    """

    temp_path = None
    normalized_audio_path = None

    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="업로드된 음성 파일이 비어 있습니다.",
            )

        original_filename = file.filename or "answer.webm"
        suffix = Path(original_filename).suffix or ".webm"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name

        # 브라우저 WebM을 Whisper가 안정적으로 읽도록 16kHz mono WAV로 변환한다.
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav",
        ) as normalized_file:
            normalized_audio_path = normalized_file.name

        extracted, duration_seconds = (
            speech_service.extract_audio_and_get_duration(
                temp_path,
                normalized_audio_path,
            )
        )

        # 너무 짧은 답변은 Whisper 호출 전에 명확하게 알려준다.
        if extracted and duration_seconds < 0.6:
            raise HTTPException(
                status_code=422,
                detail="답변 음성이 너무 짧습니다. 1초 이상 다시 답변해주세요.",
            )

        # WAV 변환에 성공하면 정규화된 파일을 사용하고,
        # 변환 실패 시에는 기존처럼 원본 파일로 한 번 시도한다.
        stt_source_path = (
            normalized_audio_path
            if extracted
            else temp_path
        )

        text = speech_service.run_free_whisper_stt(
            stt_source_path
        )

        if text.strip() == "STT 변환 실패":
            raise HTTPException(
                status_code=500,
                detail="Whisper STT 변환에 실패했습니다.",
            )

        if not text or not text.strip():
            raise HTTPException(
                status_code=422,
                detail="음성에서 텍스트를 인식하지 못했습니다. 다시 답변해주세요.",
            )

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
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

        if (
            normalized_audio_path
            and os.path.exists(normalized_audio_path)
        ):
            os.remove(normalized_audio_path)

        await file.close()