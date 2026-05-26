from fastapi import APIRouter

from app.eye_blinking.schemas.followup import FollowUpRequest, FollowUpResponse
from app.eye_blinking.services.followup_service import generate_follow_up_question

router = APIRouter(prefix="/followup", tags=["followup"])


@router.post("", response_model=FollowUpResponse)
async def followup(request: FollowUpRequest):
    """
    지원자의 답변을 기반으로 꼬리 질문을 생성하는 엔드포인트
    """
    return generate_follow_up_question(request)