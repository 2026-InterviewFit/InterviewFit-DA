from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


class FollowUpRequest(BaseModel):
    """
    Spring Boot에서 넘어오는 꼬리질문 생성 요청 DTO.

    팀원 question/interview/jobRole 코드와 섞일 수 있으므로
    camelCase와 snake_case를 모두 허용한다.
    Spring에서 interviewId/jobRoleId/questionId를 이용해 job/original_question을 정리해서 보내는 것이 원칙이지만,
    FastAPI 단독 테스트를 위해 job/original_question도 직접 받을 수 있게 둔다.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        extra="ignore",
    )

    job: str | None = Field(default=None, examples=["백엔드 개발자"])

    original_question: str | None = Field(
        default=None,
        validation_alias=AliasChoices("original_question", "originalQuestion", "question"),
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

    question_id: int | None = Field(
        default=None,
        validation_alias=AliasChoices("question_id", "questionId"),
    )

    @field_validator("answer")
    @classmethod
    def answer_must_not_be_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("answer 값이 비어 있습니다.")
        return value.strip()


class FollowUpResponse(BaseModel):
    follow_up_question: str