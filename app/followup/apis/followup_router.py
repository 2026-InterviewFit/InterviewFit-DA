from fastapi import APIRouter

from app.followup.schemas.followup_schema import FollowUpRequest, FollowUpResponse
from app.followup.services.followup_service import generate_follow_up_question

router = APIRouter(prefix="/api/followup", tags=["FollowUp"])


@router.post("/create", response_model=FollowUpResponse)
def create_follow_up_question(request: FollowUpRequest):
    follow_up_question = generate_follow_up_question(
        job=request.job,
        original_question=request.original_question,
        answer=request.answer,
        interview_style=request.interview_style,
    )

    return FollowUpResponse(follow_up_question=follow_up_question)