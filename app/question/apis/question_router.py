from fastapi import APIRouter

from app.question.schemas.question_schema import QuestionRequest, QuestionResponse

from app.question.services.question_service import generate_first_question

router = APIRouter(prefix="/api/question",tags=["Question"])


@router.post("/create", response_model=QuestionResponse)
def create_first_question(request: QuestionRequest):
    question_list = generate_first_question(
        request.count,
        request.job,
        request.resume_content
    )

    return QuestionResponse(
        questions=question_list
    )