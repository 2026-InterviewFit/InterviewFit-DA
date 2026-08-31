import asyncio
import os
import tempfile
from pathlib import Path
from typing import Callable, TypeVar

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
)
from starlette.concurrency import run_in_threadpool

from app.speech_coaching.services import speech_service
from app.speech_coaching.schemas.speech_schema import (
    SpeechAnalyzeRequest,
    SpeechTranscriptionResponse,
)


router = APIRouter()


# =========================================================
# Speech 작업 실행 정책
# =========================================================
#
# 실제 일대일 면접에서는 아래 작업이 겹친다.
#
# 1) 질문별 답변의 백그라운드 Speech/Gemini 분석
# 2) 꼬리질문 생성을 위한 즉시 STT
# 3) 꼬리질문 답변 저장을 위한 즉시 STT
#
# 하나의 semaphore를 공유하면 백그라운드 /analyze 요청이 슬롯을 모두
# 차지해 /transcribe-file이 수 분 동안 기다리는 starvation이 생길 수 있다.
#
# 따라서 두 lane을 분리한다.
# - analysis lane: 무거운 백그라운드 분석은 한 번에 1개
# - interactive STT lane: 사용자 진행에 필요한 STT는 별도로 한 번에 1개
#
# 각 blocking 작업은 thread pool에서 실행되므로 FastAPI event loop도
# Whisper/ffmpeg/librosa/Gemini 작업 때문에 막히지 않는다.
ANALYSIS_CONCURRENCY_LIMIT = 1
INTERACTIVE_STT_CONCURRENCY_LIMIT = 1

_analysis_semaphore = asyncio.Semaphore(ANALYSIS_CONCURRENCY_LIMIT)
_interactive_stt_semaphore = asyncio.Semaphore(
    INTERACTIVE_STT_CONCURRENCY_LIMIT
)

T = TypeVar("T")


async def _run_blocking_job(
    semaphore: asyncio.Semaphore,
    func: Callable[..., T],
    *args,
    **kwargs,
) -> T:
    async with semaphore:
        return await run_in_threadpool(
            func,
            *args,
            **kwargs,
        )


def _transcribe_saved_file(
    temp_path: str,
    normalized_audio_path: str,
) -> str:
    """
    업로드 파일 정규화 + Whisper STT.

    이 함수는 blocking 함수이므로 반드시
    _run_blocking_job()을 통해 thread pool에서 호출한다.
    """
    extracted, duration_seconds = (
        speech_service.extract_audio_and_get_duration(
            temp_path,
            normalized_audio_path,
        )
    )

    if extracted and duration_seconds < 0.6:
        raise HTTPException(
            status_code=422,
            detail="답변 음성이 너무 짧습니다. 1초 이상 다시 답변해주세요.",
        )

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

    return text.strip()


# =========================================================
# 전체/질문별 스피치 분석 API
# =========================================================

@router.post("/analyze")
async def analyze_interview(
    request: SpeechAnalyzeRequest,
):
    """
    전체 영상 또는 질문별 답변의 Speech/Gemini 분석.

    백그라운드 분석끼리는 한 번에 하나씩 실행한다.
    중요한 점은 이 lane과 /transcribe-file lane이 분리되어 있어
    질문별 분석 때문에 꼬리질문 STT가 큐 뒤에서 굶지 않는다는 것이다.
    """
    return await _run_blocking_job(
        _analysis_semaphore,
        speech_service.process_interview_analysis,
        source_type=request.source_type,
        source_value=request.source_value,
        job_category=request.job_category,
        interview_id=request.interview_id,
        interview_answer_id=request.interview_answer_id,
        question_id=request.question_id,
        original_question=request.original_question,
    )


# =========================================================
# 기본 질문/꼬리질문 답변 STT 전용 API
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

    사용자 인터랙션에 직접 필요한 요청이므로 백그라운드 분석과
    별도 semaphore를 사용한다.
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

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav",
        ) as normalized_file:
            normalized_audio_path = normalized_file.name

        text = await _run_blocking_job(
            _interactive_stt_semaphore,
            _transcribe_saved_file,
            temp_path,
            normalized_audio_path,
        )

        return SpeechTranscriptionResponse(
            text=text,
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
