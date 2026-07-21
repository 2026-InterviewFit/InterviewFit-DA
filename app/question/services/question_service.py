from google.genai import types

from app.common.exceptions.custom_exception import CustomException
from app.common.exceptions.error_code import ErrorCode
from app.common.exceptions.status_code import StatusCode
from app.core.config import settings
from app.core.gemini_client import question_client
from app.question.prompts.question_prompts import create_question_prompt
from app.question.schemas.question_schema import QuestionResponse, CoverLetterQuestionAnalysisDto2

def generate_first_question(
    count: int,
    job: str,
    analyses: list[CoverLetterQuestionAnalysisDto2],
    previous_questions: list[str] | None
) -> QuestionResponse:
    if question_client is None:
        raise CustomException(
            status=StatusCode.INTERNAL_SERVER_ERROR,
            code=ErrorCode.QUESTION_GENERATION_FAILED,
            message="Gemini 클라이언트가 초기화되지 않았습니다."
        )

    system_prompt, user_prompt = create_question_prompt(
        count=count,
        job=job,
        analyses=analyses,
        previous_questions=previous_questions,
    )

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0.3,
        max_output_tokens=1024,
        response_mime_type="application/json",
        response_schema=QuestionResponse,
    )

    contents = [
        {
            "role": "user",
            "parts": [
                {
                    "text": user_prompt
                }
            ]
        }
    ]

    try:
        response = question_client.models.generate_content(
            model=getattr(settings, "GEMINI_MODEL", "gemini-2.5-flash-lite"),
            contents=contents,
            config=config
        )
    except Exception:
        raise CustomException(
            status=StatusCode.INTERNAL_SERVER_ERROR,
            code=ErrorCode.QUESTION_GENERATION_FAILED,
            message="질문 생성 중 오류가 발생했습니다."
        )

    try:
        parsed: QuestionResponse = response.parsed
    except Exception:
        raise CustomException(
            status=StatusCode.INTERNAL_SERVER_ERROR,
            code=ErrorCode.INVALID_QUESTION_RESPONSE,
            message="질문 생성 결과를 파싱할 수 없습니다."
        )

    # 요청 개수보다 많이 생성된 경우 제한
    if len(parsed.questions) > count:
        parsed.questions = parsed.questions[:count]

    return parsed