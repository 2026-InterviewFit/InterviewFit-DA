from fastapi import APIRouter, HTTPException

from app.followup.schemas.followup_schema import FollowUpRequest, FollowUpResponse
from app.followup.services.followup_service import generate_follow_up_question


router = APIRouter()


@router.post("/create", response_model=FollowUpResponse)
def create_follow_up_question(request: FollowUpRequest):
    """
    Spring Boot에서 원 질문 + 사용자 답변 + 직무 정보를 받아
    Gemini 기반 꼬리질문 1개를 생성한다.

    최종 경로:
    POST /apis/followup/create
    """
    try:
        follow_up_question = generate_follow_up_question(
            job=request.job,
            original_question=request.original_question,
            answer=request.answer,
            interview_style=request.interview_style,
        )

        return FollowUpResponse(
            follow_up_question=follow_up_question,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"꼬리 질문 생성 중 오류가 발생했습니다: {str(e)}",
        )