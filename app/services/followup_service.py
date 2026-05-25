from app.schemas.followup import FollowUpRequest, FollowUpResponse

def generate_follow_up_question(request: FollowUpRequest) -> FollowUpResponse:
    """
    꼬리 질문 생성 함수 (추후 구현 예정)

    TODO:
        - Gemini/GPT API 호출하여 직무 및 답변 맥락 기반 꼬리질문 생성
        - interview_style(normal/friendly/pressure)에 따라 프롬프트 분기
    """
    pass