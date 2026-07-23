from fastapi import APIRouter, HTTPException

from app.pt_interview.schema.pt_question_schema import (
    PtQuestionGenerateRequest,
    PtQuestionGenerateResponse,
)
from app.pt_interview.service.pt_question_service import (
    PtQuestionService,
)


router = APIRouter(
    prefix="/apis/pt-question",
    tags=["PT Question"],
)

service = PtQuestionService()


@router.post(
    "/generate",
    response_model=PtQuestionGenerateResponse,
)
def generate_pt_questions(
    request: PtQuestionGenerateRequest,
):

    try:

        return service.generate_questions(
            document_summary=(
                request.document_summary
            ),
            slides_text=(
                request.slides_text
            ),
            question_count=(
                request.question_count
            ),
        )

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )