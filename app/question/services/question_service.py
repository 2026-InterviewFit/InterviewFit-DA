import json
import re
from google.genai import types
from app.common.exceptions.custom_exception import CustomException
from app.common.exceptions.error_code import ErrorCode
from app.common.exceptions.status_code import StatusCode
from app.core.gemini_client import question_client
from app.question.prompts.question_prompts import create_question_prompt

def generate_first_question(count: int, job: str, cover_letter: str | None = None):
    if question_client is None:
        raise CustomException(
            status=StatusCode.INTERNAL_SERVER_ERROR,
            code=ErrorCode.QUESTION_GENERATION_FAILED,
            message="Gemini 클라이언트가 초기화되지 않았습니다."
        )

    config = types.GenerateContentConfig(
        temperature=0.7,
        max_output_tokens=1024,
    )

    system_prompt, user_prompt = create_question_prompt(
        count=count,
        job=job,
        cover_letter=cover_letter
    )

    contents = [
        {
            "role": "user",
            "parts": [
                {
                    "text": f"{system_prompt}{user_prompt}"
                }
            ]
        }
    ]

    try:
        response = question_client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=contents,
            config=config
        )
        print(response.text)
    except Exception:
        raise CustomException(
            status=StatusCode.INTERNAL_SERVER_ERROR,
            code=ErrorCode.QUESTION_GENERATION_FAILED,
            message="질문 생성 중 오류가 발생했습니다."
        )

    try:
        return safe_parse(response.text)
    except ValueError:
        raise CustomException(
            status=StatusCode.INTERNAL_SERVER_ERROR,
            code=ErrorCode.INVALID_QUESTION_RESPONSE,
            message="질문 생성 결과를 파싱할 수 없습니다."
        )

def safe_parse(text: str):
    try:
        text = text.strip()

        match = re.search(r"\[.*?\]", text, re.DOTALL)
        if match:
            return json.loads(match.group())

        return json.loads(text)
    except json.JSONDecodeError:
        raise ValueError("면접 질문 리스트를 찾을 수 없습니다.")