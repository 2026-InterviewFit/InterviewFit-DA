import logging

import httpx
from pydantic import ValidationError

from app.core.config import settings
from app.speech_coaching.schemas.speech_schema import ExternalAiAnalysis


logger = logging.getLogger(__name__)


def analyze_interview_answer(
    question: str | None,
    answer: str | None,
) -> ExternalAiAnalysis | None:
    """외부 AI로 답변을 분석한다. 실패는 기존 분석 흐름에 전파하지 않는다."""
    base_url = settings.AI_SERVER_URL.strip()
    if not base_url:
        logger.info("외부 면접 답변 평가 AI URL이 설정되지 않아 호출을 건너뜁니다.")
        return None

    safe_question = (question or "").strip()
    safe_answer = (answer or "").strip()
    endpoint = f"{base_url.rstrip('/')}/analyze"
    timeout = httpx.Timeout(
        connect=settings.AI_SERVER_CONNECT_TIMEOUT_SECONDS,
        read=settings.AI_SERVER_READ_TIMEOUT_SECONDS,
        write=settings.AI_SERVER_READ_TIMEOUT_SECONDS,
        pool=settings.AI_SERVER_CONNECT_TIMEOUT_SECONDS,
    )

    try:
        response = httpx.post(
            endpoint,
            json={
                "question": safe_question,
                "answer": safe_answer,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        return ExternalAiAnalysis.model_validate(response.json())
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "외부 면접 답변 평가 AI가 오류 상태를 반환했습니다: status=%s",
            exc.response.status_code,
        )
    except httpx.TimeoutException as exc:
        logger.warning(
            "외부 면접 답변 평가 AI 호출 시간이 초과됐습니다: error_type=%s",
            type(exc).__name__,
        )
    except httpx.RequestError as exc:
        logger.warning(
            "외부 면접 답변 평가 AI 연결에 실패했습니다: error_type=%s",
            type(exc).__name__,
        )
    except (ValueError, ValidationError) as exc:
        logger.warning(
            "외부 면접 답변 평가 AI 응답 검증에 실패했습니다: error_type=%s",
            type(exc).__name__,
        )
    except Exception as exc:
        # 외부 클라이언트의 예기치 않은 오류도 기존 Gemini 분석을 중단시키지 않는다.
        logger.warning(
            "외부 면접 답변 평가 AI 처리에 실패했습니다: error_type=%s",
            type(exc).__name__,
        )

    return None
