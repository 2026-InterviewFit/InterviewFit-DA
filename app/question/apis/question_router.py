from fastapi import APIRouter
from app.question.schemas.question_schema import QuestionRequest, QuestionResponse
from app.question.services.question_service import generate_first_question

router = APIRouter()

@router.post("/create", response_model=QuestionResponse)
def create_first_question(request: QuestionRequest):
    question_list = generate_first_question(
        count=request.count,
        job=request.job,
        cover_letter=request.cover_letter,
        previous_questions=request.previous_questions
    )
    return QuestionResponse(questions=question_list)