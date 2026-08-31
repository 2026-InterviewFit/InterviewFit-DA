import logging

from google.genai import types
from pydantic import ValidationError

from app.common.exceptions.custom_exception import CustomException
from app.common.exceptions.error_code import ErrorCode
from app.common.exceptions.status_code import StatusCode
from app.core.config import settings
from app.core.gemini_client import create_gemini_client
from app.group_interview.prompts.comparison_prompts import SYSTEM_PROMPT, create_comparison_prompt
from app.group_interview.schemas.comparison_schema import (
    GroupComparisonFeedbackRequest,
    GroupComparisonFeedbackResponse,
)

logger = logging.getLogger(__name__)

comparison_client = create_gemini_client()


def _feedback_error(message: str, cause: Exception | None = None) -> CustomException:
    exception = CustomException(
        status=StatusCode.INTERNAL_SERVER_ERROR,
        code=ErrorCode.GROUP_COMPARISON_FEEDBACK_FAILED,
        message=message,
    )
    if cause is not None:
        exception.__cause__ = cause
    return exception


def _validate_response_participants(
    request: GroupComparisonFeedbackRequest,
    response: GroupComparisonFeedbackResponse,
) -> None:
    requested_ids = [item.participantUserId for item in request.participants]
    response_ids = [item.participantUserId for item in response.participants]

    if len(response_ids) != len(set(response_ids)):
        raise ValueError("Gemini 응답에 중복된 participantUserId가 있습니다.")
    if set(response_ids) != set(requested_ids):
        raise ValueError("Gemini 응답의 참가자 집합이 요청과 일치하지 않습니다.")


def generate_comparison_feedback(
    request: GroupComparisonFeedbackRequest,
) -> GroupComparisonFeedbackResponse:
    if comparison_client is None:
        raise _feedback_error("Gemini 클라이언트가 초기화되지 않았습니다.")

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.3,
        max_output_tokens=2048,
        response_mime_type="application/json",
        response_schema=GroupComparisonFeedbackResponse,
    )

    try:
        gemini_response = comparison_client.models.generate_content(
            model=getattr(settings, "GEMINI_MODEL", "gemini-3.5-flash-lite"),
            contents=create_comparison_prompt(request),
            config=config,
        )
    except Exception as exc:
        logger.exception("그룹 면접 비교 피드백 Gemini 호출 실패")
        raise _feedback_error("비교 피드백 생성 중 오류가 발생했습니다.", exc)

    try:
        parsed = gemini_response.parsed
        if isinstance(parsed, GroupComparisonFeedbackResponse):
            result = parsed
        else:
            result = GroupComparisonFeedbackResponse.model_validate(parsed)
        _validate_response_participants(request, result)
        return result
    except (AttributeError, TypeError, ValueError, ValidationError) as exc:
        logger.warning(
            "그룹 면접 비교 피드백 응답 검증 실패: error_type=%s",
            type(exc).__name__,
        )
        raise _feedback_error("비교 피드백 결과 형식이 올바르지 않습니다.", exc)
