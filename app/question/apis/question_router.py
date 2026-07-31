from fastapi import APIRouter

from app.question.schemas.question_schema import QuestionRequest, QuestionResponse
from app.question.services.question_service import generate_first_question

router = APIRouter()

@router.post("/one-to-one/create", response_model=QuestionResponse)
async def create_first_question(request: QuestionRequest):
    return await generate_first_question(
        count=request.count,
        job=request.job,
        analyses=request.analyses,
        previous_questions=request.previous_questions,
        reusable_questions=request.reusable_questions,
    )