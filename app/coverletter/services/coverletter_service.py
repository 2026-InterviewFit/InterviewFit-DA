import logging
import asyncio

from google.genai import types

from app.common.exceptions.custom_exception import CustomException
from app.common.exceptions.error_code import ErrorCode
from app.common.exceptions.status_code import StatusCode
from app.core.config import settings
from app.core.gemini_client import coverletter_client
from app.coverletter.prompts.coverletter_prompts import (
    create_coverletter_extraction_prompt,
    create_coverletter_analysis_prompt,
    create_job_recommendation_prompt, create_coverletter_importance_prompt
)
from app.coverletter.schemas.coverletter_schema import (
    CoverLetterQuestionItem,
    CoverLetterQuestionResponse,
    CoverLetterQuestionAnalysisItem,
    JobRecommendation,
    CoverLetterDetailAnalysisResponse,
    CoverLetterQuestionAnalysisResult, CoverLetterImportanceResponse
)

logger = logging.getLogger(__name__)

async def generate_cover_letter_analysis(coverletter: str) -> CoverLetterDetailAnalysisResponse:
    question_response = await generate_coverletter_questions(coverletter)

    importance_response  = await generate_coverletter_question_importance(question_response.cover_letter_questions)

    importance_map = {
        item.question_number: item.importance_score
        for item in importance_response.question_importance_scores
    }

    analyses = []

    for question in question_response.cover_letter_questions:
        analysis = await generate_coverletter_question_analysis(item=question)
        analyses.append(
            CoverLetterQuestionAnalysisItem(
                question_number=question.question_number,
                question=question.question,
                answer=question.answer,
                summary=question.summary,
                star=analysis.star,
                strengths=analysis.strengths,
                improvements=analysis.improvements,
                skills=analysis.skills,
                importance_score=importance_map[
                    question.question_number
                ],
            )
        )

    job = await generate_job_recommendation(coverletter)

    return CoverLetterDetailAnalysisResponse(
        questions_analysis=analyses,
        job=job,
    )

async def _generate(system_prompt: str, user_prompt: str, response_schema):
    if coverletter_client is None:
        raise CustomException(
            status=StatusCode.INTERNAL_SERVER_ERROR,
            code=ErrorCode.COVERLETTER_ANALYSIS_FAILED,
            message="Gemini 클라이언트가 초기화되지 않았습니다."
        )

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0.3,
        max_output_tokens=2048,
        response_mime_type="application/json",
        response_schema=response_schema,
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

    max_retry = 1

    for attempt in range(max_retry + 1):
        try:
            response = await coverletter_client.aio.models.generate_content(
                model=getattr(settings, "GEMINI_MODEL", "gemini-2.5-flash-lite"),
                contents=contents,
                config=config,
            )

            if response.parsed is None:
                raise ValueError("Gemini parsed response is None")

            return response.parsed
        except Exception as e:
            if attempt == max_retry:
                logger.exception(
                    "Gemini generate 최종 실패 - attempts=%d",
                    max_retry + 1,
                )

                raise CustomException(
                    status=StatusCode.INTERNAL_SERVER_ERROR,
                    code=ErrorCode.COVERLETTER_ANALYSIS_FAILED,
                    message="Gemini 호출 중 오류가 발생했습니다."
                ) from e

            logger.warning(
                "Gemini generate 실패, 재시도 - attempt=%d/%d, "
                "error_type=%s, error=%s",
                attempt + 1,
                max_retry + 1,
                type(e).__name__,
                str(e),
            )
            await asyncio.sleep(1)

    return None


async def generate_coverletter_questions(coverletter: str) -> CoverLetterQuestionResponse:
    system_prompt, user_prompt = create_coverletter_extraction_prompt(coverletter)

    return await _generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_schema=CoverLetterQuestionResponse,
    )

async def generate_coverletter_question_importance(questions: list[CoverLetterQuestionItem]) -> CoverLetterImportanceResponse:
    system_prompt, user_prompt = create_coverletter_importance_prompt(questions)

    return await _generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_schema=CoverLetterImportanceResponse,
    )

async def generate_coverletter_question_analysis(item: CoverLetterQuestionItem) -> CoverLetterQuestionAnalysisResult:
    system_prompt, user_prompt = create_coverletter_analysis_prompt(
        question=item.question,
        answer=item.answer,
        summary=item.summary,
    )

    return await _generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_schema=CoverLetterQuestionAnalysisResult,
    )

async def generate_job_recommendation(coverletter: str) -> JobRecommendation:
    system_prompt, user_prompt = create_job_recommendation_prompt(coverletter)

    return await _generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_schema=JobRecommendation,
    )