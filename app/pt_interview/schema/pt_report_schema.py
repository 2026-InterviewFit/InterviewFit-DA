from pydantic import BaseModel, Field


class PtReportGenerateRequest(BaseModel):
    presentation_summary: str = ""

    questions: list[str] = Field(default_factory=list)
    answers: list[str] = Field(default_factory=list)

    eye_scores: list[float] = Field(default_factory=list)
    blink_scores: list[float] = Field(default_factory=list)
    pose_scores: list[float] = Field(default_factory=list)
    speech_scores: list[float] = Field(default_factory=list)


class PtReportGenerateResponse(BaseModel):
    content_score: int = Field(ge=0, le=100)
    structure_score: int = Field(ge=0, le=100)
    delivery_score: int = Field(ge=0, le=100)
    qa_score: int = Field(ge=0, le=100)
    overall_score: int = Field(ge=0, le=100)

    presentation_summary: str
    strengths: str
    weaknesses: str
    improvement: str