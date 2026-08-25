import json
import logging
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError

from app.common.exceptions.custom_exception import CustomException
from app.common.exceptions.error_code import ErrorCode
from app.common.exceptions.status_code import StatusCode
from app.core.config import settings
from app.speech_coaching.schemas.speech_schema import AiEvaluation

logger = logging.getLogger(__name__)

client = genai.Client(api_key=settings.GEMINI_API_KEY)

_RAW_RESPONSE_LOG_LIMIT = 300


class GeminiStarScore(BaseModel):
    """Fixed object schema supported by the Gemini Developer API."""

    S: int = Field(ge=0, le=10)
    T: int = Field(ge=0, le=10)
    A: int = Field(ge=0, le=10)
    R: int = Field(ge=0, le=10)


class GeminiAiEvaluation(BaseModel):
    """Gemini-only schema; the public response continues to use AiEvaluation."""

    star_score: GeminiStarScore = Field(
        description="지원자가 실제로 말한 답변의 STAR 요소별 평가 점수",
    )
    feedback: str = Field(
        description="지원자가 실제로 말한 답변에 대한 평가와 구체적인 개선점",
    )
    keywords: list[str] = Field(
        description="지원자가 실제로 말한 답변에서 추출한 핵심 키워드",
    )
    model_answer: str = Field(
        description=(
            "면접 질문 원문에 직접 답하는 1인칭 모범 면접 답변 예시. "
            "평가나 작성 지침이 아닌 지원자가 실제로 말할 수 있는 완성된 답변"
        ),
    )


def _ai_evaluation_error(message: str, cause: Exception | None = None) -> CustomException:
    exception = CustomException(
        status=StatusCode.INTERNAL_SERVER_ERROR,
        code=ErrorCode.AI_EVALUATION_FAILED,
        message=message,
    )
    if cause is not None:
        exception.__cause__ = cause
    return exception


def _parse_json_object(raw_text: str) -> dict[str, Any]:
    """Extract the first JSON object, tolerating fences, prose and raw newlines."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError("Gemini 응답이 비어 있습니다.")

    text = raw_text.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```JSON").removeprefix("```")
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        text = text.strip()

    # strict=False permits literal control characters such as line breaks in a
    # JSON string. raw_decode permits explanatory text after the JSON object.
    decoder = json.JSONDecoder(strict=False)
    last_error: json.JSONDecodeError | None = None
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text, index)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(value, dict):
            return value

    if last_error is not None:
        raise last_error
    raise ValueError("Gemini 응답에서 JSON 객체를 찾을 수 없습니다.")


def _validate_ai_evaluation(value: Any) -> dict[str, Any]:
    if isinstance(value, GeminiAiEvaluation):
        evaluation = AiEvaluation.model_validate(value.model_dump())
    elif isinstance(value, AiEvaluation):
        evaluation = value
    else:
        if not isinstance(value, dict):
            raise TypeError(f"Gemini 평가 결과가 객체가 아닙니다: {type(value).__name__}")
        evaluation = AiEvaluation.model_validate(value)

    required_star_keys = {"S", "T", "A", "R"}
    actual_star_keys = set(evaluation.star_score)
    if actual_star_keys != required_star_keys:
        missing = sorted(required_star_keys - actual_star_keys)
        unexpected = sorted(actual_star_keys - required_star_keys)
        raise ValueError(
            f"star_score 키가 올바르지 않습니다. missing={missing}, unexpected={unexpected}"
        )
    if any(score < 0 or score > 10 for score in evaluation.star_score.values()):
        raise ValueError("star_score는 0에서 10 사이의 정수여야 합니다.")

    return evaluation.model_dump(by_alias=True)


def evaluate_interview_content(
    script: str,
    job_category: str,
    original_question: str | None = None,
) -> dict[str, Any]:
    question = (original_question or "").strip()
    answer = (script or "").strip()
    prompt = f"""
    당신은 {job_category} 직무의 전문 면접관입니다.

    아래 네 출력 필드의 역할을 엄격히 구분하세요.
    1. star_score: 지원자가 실제로 말한 답변만 평가합니다. S/T/A/R을 각각 0~10 정수로 채웁니다.
    2. feedback: 지원자가 실제로 말한 답변에 대한 평가와 구체적인 개선점입니다.
       답변이 비어 있으면 답변이 없어 평가할 수 없다고 명시합니다.
    3. keywords: 지원자가 실제로 말한 답변의 핵심 키워드입니다. 답변이 비어 있으면 빈 배열입니다.
    4. model_answer: 아래 '면접 질문 원문'에 직접 답하는 완성된 모범 면접 답변 예시입니다.
       지원자가 면접에서 그대로 말할 수 있는 자연스러운 1인칭 답변으로 작성하세요.
       사용자 답변이 부족하거나 비어 있어도 질문 원문을 바탕으로 구체적인 모범 답변을 작성하세요.

    model_answer에는 평가, 개선 조언, 답변 작성법을 넣지 마세요.
    특히 'STAR 방식으로 답변해주세요', '구체적인 경험이 부족합니다',
    '현재 답변은 평가하기 어렵습니다', '~하는 것이 좋습니다' 같은 코칭 문장을 넣지 마세요.
    이러한 평가는 feedback에만 작성하세요.

    직무: {job_category}
    면접 질문 원문: {question if question else "제공되지 않음"}
    지원자가 실제로 말한 답변: {answer if answer else "(빈 답변)"}
    """

    config = types.GenerateContentConfig(
        temperature=0.3,
        max_output_tokens=2048,
        response_mime_type="application/json",
        response_schema=GeminiAiEvaluation,
    )

    try:
        response = client.models.generate_content(
            model=getattr(settings, "GEMINI_MODEL", "gemini-3.5-flash-lite"),
            contents=prompt,
            config=config,
        )
    except Exception as exc:
        logger.exception("Gemini 평가 API 호출 실패")
        raise _ai_evaluation_error("Gemini 평가 중 오류가 발생했습니다.", exc)

    parsed = getattr(response, "parsed", None)
    if parsed is None:
        raw_text = getattr(response, "text", "")
        try:
            parsed = _parse_json_object(raw_text)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            # repr exposes control characters; the bound avoids logging an
            # entire response that may contain user-derived information.
            preview = raw_text[:_RAW_RESPONSE_LOG_LIMIT] if isinstance(raw_text, str) else raw_text
            logger.error(
                "Gemini 평가 응답 JSON 파싱 실패: raw_preview=%r raw_length=%s",
                preview,
                len(raw_text) if isinstance(raw_text, str) else "unknown",
            )
            raise _ai_evaluation_error("Gemini 평가 결과를 파싱할 수 없습니다.", exc)

    try:
        return _validate_ai_evaluation(parsed)
    except (TypeError, ValueError, ValidationError) as exc:
        # ValidationError may embed portions of the model output, so do not log
        # its full message here.
        logger.warning(
            "Gemini 평가 응답 스키마 검증 실패: error_type=%s",
            type(exc).__name__,
        )
        raise _ai_evaluation_error("Gemini 평가 결과 형식이 올바르지 않습니다.", exc)
