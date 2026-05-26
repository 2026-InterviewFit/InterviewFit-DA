from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class FollowUpRequest(BaseModel):
    """
    Spring Boot에서 넘어오는 camelCase와
    FastAPI 내부 snake_case를 모두 허용한다.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        extra="ignore",
    )

    job: str = Field(..., examples=["백엔드 개발자"])

    original_question: str = Field(
        ...,
        validation_alias=AliasChoices("original_question", "originalQuestion"),
        examples=["프로젝트에서 가장 어려웠던 문제를 어떻게 해결했나요?"],
    )

    answer: str = Field(
        ...,
        examples=[
            "팀 프로젝트에서 API 응답 속도가 느린 문제가 있었고, "
            "쿼리 최적화와 캐싱을 적용해서 해결했습니다."
        ],
    )

    interview_style: str = Field(
        default="normal",
        validation_alias=AliasChoices("interview_style", "interviewStyle"),
        examples=["normal", "friendly", "pressure"],
        description="normal: 일반 면접, friendly: 부드러운 면접, pressure: 압박 면접",
    )

    user_id: int | None = Field(
        default=None,
        validation_alias=AliasChoices("user_id", "userId"),
    )

    interview_id: int | None = Field(
        default=None,
        validation_alias=AliasChoices("interview_id", "interviewId"),
    )


class FollowUpResponse(BaseModel):
    follow_up_question: str