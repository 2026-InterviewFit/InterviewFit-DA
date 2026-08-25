from pydantic import BaseModel, Field, field_validator, model_validator


class ParticipantComparisonScore(BaseModel):
    participantUserId: int
    contentScore: float = Field(ge=0, le=100)
    speechScore: float = Field(ge=0, le=100)
    gazeScore: float = Field(ge=0, le=100)
    poseScore: float = Field(ge=0, le=100)
    blinkScore: float = Field(ge=0, le=100)
    totalScore: float = Field(ge=0, le=100)
    rank: int = Field(ge=1)


class GroupComparisonFeedbackRequest(BaseModel):
    roomId: int
    questionIndex: int | None = Field(default=None, ge=0, le=2)
    participants: list[ParticipantComparisonScore] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_unique_participants(self):
        participant_ids = [item.participantUserId for item in self.participants]
        if len(participant_ids) != len(set(participant_ids)):
            raise ValueError("participantUserId는 중복될 수 없습니다.")
        return self


class ParticipantComparisonFeedback(BaseModel):
    participantUserId: int
    strengthSummary: str = Field(min_length=1)
    improvementSummary: str = Field(min_length=1)

    @field_validator("strengthSummary", "improvementSummary")
    @classmethod
    def validate_non_blank_summary(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("피드백 문장은 비어 있을 수 없습니다.")
        return value.strip()


class GroupComparisonFeedbackResponse(BaseModel):
    participants: list[ParticipantComparisonFeedback]
