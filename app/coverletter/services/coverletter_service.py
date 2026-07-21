from google.genai import types

from app.common.exceptions.custom_exception import CustomException
from app.common.exceptions.error_code import ErrorCode
from app.common.exceptions.status_code import StatusCode
from app.core.config import settings
from app.core.gemini_client import coverletter_client
from app.coverletter.prompts.coverletter_prompts import create_analysis_prompt
from app.coverletter.schemas.coverletter_schema import CoverLetterAnalysisResponse

def generate_cover_letter_analysis(coverletter: str) -> CoverLetterAnalysisResponse:
    if coverletter_client is None:
        raise CustomException(
            status=StatusCode.INTERNAL_SERVER_ERROR,
            code=ErrorCode.COVERLETTER_ANALYSIS_FAILED,
            message="Gemini 클라이언트가 초기화되지 않았습니다."
        )

    system_prompt, user_prompt = create_analysis_prompt(
        coverletter=coverletter,
    )

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0.3,
        max_output_tokens=2048,
        response_mime_type="application/json",
        response_schema=CoverLetterAnalysisResponse,
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
        response = coverletter_client.models.generate_content(
            model=getattr(settings, "GEMINI_MODEL", "gemini-2.5-flash-lite"),
            contents=contents,
            config=config,
        )
    except Exception:
        raise CustomException(
            status=StatusCode.INTERNAL_SERVER_ERROR,
            code=ErrorCode.COVERLETTER_ANALYSIS_FAILED,
            message="자소서 분석 중 오류가 발생했습니다."
        )

    try:
        parsed: CoverLetterAnalysisResponse = response.parsed
        return parsed
    except Exception:
        raise CustomException(
            status=StatusCode.INTERNAL_SERVER_ERROR,
            code=ErrorCode.INVALID_COVERLETTER_ANALYSIS_RESPONSE,
            message="자소서 분석 결과를 파싱하거나 검증할 수 없습니다."
        )
