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
from app.speech_coaching.schemas.speech_schema import AiEvaluation, ExternalAiAnalysis

logger = logging.getLogger(__name__)

client = genai.Client(api_key=settings.GEMINI_API_KEY)

_RAW_RESPONSE_LOG_LIMIT = 300


class GeminiStarScore(BaseModel):
    """Gemini STAR 점수 스키마."""

    S: int = Field(
        ge=0,
        le=10,
        description="Situation 점수. 0~10점",
    )

    T: int = Field(
        ge=0,
        le=10,
        description="Task 점수. 0~10점",
    )

    A: int = Field(
        ge=0,
        le=10,
        description="Action 점수. 0~10점",
    )

    R: int = Field(
        ge=0,
        le=10,
        description="Result 점수. 0~10점",
    )


class GeminiAiEvaluation(BaseModel):
    """Gemini AI 답변 평가 결과 스키마."""

    star_score: GeminiStarScore = Field(
        description="지원자가 실제로 말한 답변의 STAR 요소별 평가 점수",
    )

    feedback: str = Field(
        description="지원자가 실제로 말한 답변에 대한 구체적인 평가와 개선점",
    )

    keywords: list[str] = Field(
        description="지원자의 실제 답변에서 추출한 핵심 키워드",
    )

    model_answer: str = Field(
        description=(
            "면접 질문 자체에 직접 답하는 자연스러운 1인칭 모범 면접 답변. "
            "평가나 개선 조언이 아니라 실제 면접에서 말할 수 있는 완성된 답변"
        ),
    )


def _ai_evaluation_error(
    message: str,
    cause: Exception | None = None,
) -> CustomException:
    """
    AI 평가 관련 공통 예외 생성.
    """

    exception = CustomException(
        status=StatusCode.INTERNAL_SERVER_ERROR,
        code=ErrorCode.AI_EVALUATION_FAILED,
        message=message,
    )

    if cause is not None:
        exception.__cause__ = cause

    return exception


def _parse_json_object(
    raw_text: str,
) -> dict[str, Any]:
    """
    Gemini 텍스트 응답에서 JSON 객체를 추출한다.

    다음 형태를 모두 허용한다.
    - 일반 JSON
    - ```json 코드 블록
    - JSON 앞뒤에 설명 문장이 포함된 경우
    - JSON 문자열 내부의 raw newline
    """

    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError(
            "Gemini 응답이 비어 있습니다."
        )

    text = raw_text.strip()

    if text.startswith("```"):
        text = (
            text
            .removeprefix("```json")
            .removeprefix("```JSON")
            .removeprefix("```")
        )

        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]

        text = text.strip()

    decoder = json.JSONDecoder(
        strict=False
    )

    last_error: json.JSONDecodeError | None = None

    for index, character in enumerate(text):

        if character != "{":
            continue

        try:
            value, _ = decoder.raw_decode(
                text,
                index,
            )

        except json.JSONDecodeError as exc:
            last_error = exc
            continue

        if isinstance(value, dict):
            return value

    if last_error is not None:
        raise last_error

    raise ValueError(
        "Gemini 응답에서 JSON 객체를 찾을 수 없습니다."
    )


def _validate_ai_evaluation(
    value: Any,
) -> dict[str, Any]:
    """
    Gemini 응답을 기존 AiEvaluation 스키마로 검증한다.
    """

    if isinstance(
        value,
        GeminiAiEvaluation,
    ):
        evaluation = AiEvaluation.model_validate(
            value.model_dump()
        )

    elif isinstance(
        value,
        AiEvaluation,
    ):
        evaluation = value

    else:

        if not isinstance(
            value,
            dict,
        ):
            raise TypeError(
                "Gemini 평가 결과가 객체가 아닙니다: "
                f"{type(value).__name__}"
            )

        evaluation = AiEvaluation.model_validate(
            value
        )

    required_star_keys = {
        "S",
        "T",
        "A",
        "R",
    }

    actual_star_keys = set(
        evaluation.star_score
    )

    if actual_star_keys != required_star_keys:

        missing = sorted(
            required_star_keys
            - actual_star_keys
        )

        unexpected = sorted(
            actual_star_keys
            - required_star_keys
        )

        raise ValueError(
            "star_score 형식이 올바르지 않습니다. "
            f"missing={missing}, "
            f"unexpected={unexpected}"
        )

    if any(
        score < 0 or score > 10
        for score in evaluation.star_score.values()
    ):
        raise ValueError(
            "star_score는 0에서 10 사이의 정수여야 합니다."
        )

    return evaluation.model_dump(
        by_alias=True
    )


def _extract_finish_reason(
    response: Any,
) -> str | None:
    """
    Gemini 응답이 비어 있을 때
    첫 번째 candidate의 finish_reason을 확인한다.
    """

    try:

        candidates = getattr(
            response,
            "candidates",
            None,
        )

        if not candidates:
            return None

        candidate = candidates[0]

        finish_reason = getattr(
            candidate,
            "finish_reason",
            None,
        )

        if finish_reason is None:
            return None

        return str(
            finish_reason
        )

    except Exception:
        return None


def _extract_candidate_count(
    response: Any,
) -> int:
    """
    Gemini 응답 candidate 개수 확인.
    """

    try:

        candidates = getattr(
            response,
            "candidates",
            None,
        )

        if not candidates:
            return 0

        return len(
            candidates
        )

    except Exception:
        return 0


def _extract_response_text(
    response: Any,
) -> str | None:
    """
    response.text가 정상 문자열인지 안전하게 확인한다.
    """

    try:

        text = getattr(
            response,
            "text",
            None,
        )

        if (
            isinstance(text, str)
            and text.strip()
        ):
            return text

    except Exception:
        pass

    return None


def evaluate_interview_content(
    script: str,
    job_category: str,
    original_question: str | None = None,
    external_analysis: ExternalAiAnalysis | None = None,
) -> dict[str, Any]:
    """
    사용자의 면접 답변을 Gemini로 평가한다.

    반환 항목:
    - star_score
    - feedback
    - keywords
    - model_answer
    """

    question = (
        original_question or ""
    ).strip()

    answer = (
        script or ""
    ).strip()

    external_analysis_section = ""

    if external_analysis is not None:
        external_analysis_json = json.dumps(
            external_analysis.model_dump(mode="json"),
            ensure_ascii=False,
        )

        external_analysis_section = f"""
[외부 면접 답변 평가 AI의 참고 분석]
{external_analysis_json}

참고 분석 사용 원칙:
1. 외부 AI 결과는 평가를 돕는 참고 자료일 뿐입니다.
2. 원본 면접 질문과 지원자가 실제로 말한 답변을 최우선 근거로 판단하세요.
3. 외부 AI 결과를 그대로 복사하지 말고 직접 종합하여 평가하세요.
4. 외부 AI 결과가 원본 질문 또는 실제 답변과 충돌하면 원본 질문과 실제 답변을 우선하세요.
"""

    prompt = f"""
당신은 {job_category} 직무의 전문 면접관입니다.

아래의 면접 질문과 지원자가 실제로 말한 답변을 바탕으로
답변 내용을 평가하세요.

반드시 다음 4개 항목을 반환하세요.

1. star_score

지원자가 실제로 말한 답변만을 기준으로
STAR 구조를 평가하세요.

각 항목은 0점부터 10점 사이의 정수입니다.

S: Situation
- 상황과 배경이 구체적으로 설명되어 있는지 평가합니다.

T: Task
- 지원자가 맡았던 역할, 목표 또는 해결해야 했던 과제가
  명확하게 드러나는지 평가합니다.

A: Action
- 지원자가 실제로 어떤 행동을 했는지,
  본인의 행동과 판단이 구체적으로 설명되어 있는지 평가합니다.

R: Result
- 행동의 결과, 성과 또는 배운 점이
  구체적으로 설명되어 있는지 평가합니다.

반드시 S, T, A, R 네 항목을 모두 반환하세요.


2. feedback

지원자가 실제로 말한 답변에 대해
구체적인 평가와 개선점을 작성하세요.

좋았던 부분과 부족한 부분을 함께 설명하되,
실제 면접에서 어떻게 개선하면 좋은지
구체적으로 작성하세요.

답변이 비어 있다면
답변이 없어 평가할 수 없다고 명시하세요.


3. keywords

지원자가 실제로 말한 답변에서
핵심 키워드를 추출하세요.

답변에 실제로 등장했거나
답변 내용을 직접적으로 요약할 수 있는 키워드만 사용하세요.

답변이 비어 있다면 빈 배열을 반환하세요.


4. model_answer

아래의 면접 질문 자체에 직접 답하는
완성된 모범 면접 답변을 작성하세요.

지원자가 실제 면접에서 그대로 말할 수 있도록
자연스러운 1인칭 답변으로 작성하세요.

지원자의 기존 답변에 좋은 내용이 있다면
그 내용을 활용할 수 있지만,
부족하거나 비어 있는 부분은 질문과 직무를 바탕으로
자연스럽게 보완하세요.

model_answer에는 다음 내용을 넣지 마세요.

- 현재 답변에 대한 평가
- 답변에 대한 개선 조언
- 답변 작성 방법 설명
- "STAR 방식으로 답변하세요" 같은 지시 문장
- "구체적인 경험이 부족합니다" 같은 평가 문장
- "이렇게 답변하는 것이 좋습니다" 같은 코칭 문장

model_answer는 오직 실제 면접에서 말할 수 있는
완성된 답변 문장만 작성하세요.


직무:
{job_category}

면접 질문:
{question if question else "제공되지 않음"}

지원자가 실제로 말한 답변:
{answer if answer else "(빈 답변)"}

{external_analysis_section}
"""

    config = types.GenerateContentConfig(
        temperature=0.3,
        max_output_tokens=2048,
        response_mime_type="application/json",
        response_schema=GeminiAiEvaluation,
    )

    model_name = (
        getattr(
            settings,
            "GEMINI_MODEL",
            None,
        )
        or "gemini-3.5-flash-lite"
    )

    logger.info(
        "Gemini AI 평가 요청: "
        "model=%s answer_length=%d question_length=%d external_analysis=%s",
        model_name,
        len(answer),
        len(question),
        external_analysis is not None,
    )

    try:

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )

    except Exception as exc:

        logger.exception(
            "Gemini 평가 API 호출 실패"
        )

        raise _ai_evaluation_error(
            "Gemini 평가 중 오류가 발생했습니다.",
            exc,
        )

    parsed = getattr(
        response,
        "parsed",
        None,
    )

    # response.parsed가 있으면
    # Gemini SDK가 이미 구조화 응답을 파싱한 상태다.
    if parsed is None:

        raw_text = _extract_response_text(
            response
        )

        # parsed도 없고 text도 없는 경우
        if raw_text is None:

            finish_reason = _extract_finish_reason(
                response
            )

            candidate_count = _extract_candidate_count(
                response
            )

            logger.error(
                "Gemini 평가 응답이 비어 있음: "
                "model=%s finish_reason=%s candidates=%d",
                model_name,
                finish_reason,
                candidate_count,
            )

            raise _ai_evaluation_error(
                "Gemini 평가 응답이 비어 있습니다."
            )

        try:

            parsed = _parse_json_object(
                raw_text
            )

        except (
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:

            preview = (
                raw_text[
                    :_RAW_RESPONSE_LOG_LIMIT
                ]
                if isinstance(
                    raw_text,
                    str,
                )
                else raw_text
            )

            logger.error(
                "Gemini 평가 응답 JSON 파싱 실패: "
                "raw_preview=%r raw_length=%s",
                preview,
                (
                    len(raw_text)
                    if isinstance(
                        raw_text,
                        str,
                    )
                    else "unknown"
                ),
            )

            raise _ai_evaluation_error(
                "Gemini 평가 결과를 파싱할 수 없습니다.",
                exc,
            )

    try:

        result = _validate_ai_evaluation(
            parsed
        )

        logger.info(
            "Gemini AI 평가 성공: "
            "model=%s star_score=%s keywords_count=%d external_analysis=%s",
            model_name,
            result.get(
                "star_score"
            ),
            len(
                result.get(
                    "keywords",
                    [],
                )
            ),
            external_analysis is not None,
        )

        return result

    except (
        TypeError,
        ValueError,
        ValidationError,
    ) as exc:

        logger.warning(
            "Gemini 평가 응답 스키마 검증 실패: "
            "error_type=%s",
            type(exc).__name__,
        )

        raise _ai_evaluation_error(
            "Gemini 평가 결과 형식이 올바르지 않습니다.",
            exc,
        )

