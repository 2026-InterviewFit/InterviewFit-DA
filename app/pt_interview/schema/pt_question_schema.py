from typing import List

from pydantic import BaseModel, Field


class PtQuestionGenerateRequest(BaseModel):

    document_summary: str

    slides_text: str

    question_count: int = Field(
        default=5,
        ge=1,
        le=10,
    )


class PtQuestionResult(BaseModel):

    question_number: int

    question: str

    reason: str | None = None


class PtQuestionGenerateResponse(BaseModel):

    questions: List[PtQuestionResult]