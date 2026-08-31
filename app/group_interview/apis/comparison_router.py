from fastapi import APIRouter

from app.group_interview.schemas.comparison_schema import (
    GroupComparisonFeedbackRequest,
    GroupComparisonFeedbackResponse,
)
from app.group_interview.services.comparison_service import generate_comparison_feedback

router = APIRouter()


@router.post("/feedback", response_model=GroupComparisonFeedbackResponse)
def create_group_comparison_feedback(
    request: GroupComparisonFeedbackRequest,
) -> GroupComparisonFeedbackResponse:
    return generate_comparison_feedback(request)
