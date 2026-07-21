from pydantic import BaseModel, Field, field_validator

class StarDto(BaseModel):
    situation: str
    task: str
    action: str
    result: str

class CoverLetterQuestionAnalysisDto2(BaseModel):
    question_number: int
    summary: str
    star: StarDto = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    importance_score: float | None = None

class QuestionRequest(BaseModel):
    count: int
    job: str
    analyses: list[CoverLetterQuestionAnalysisDto2] = Field(default_factory=list)
    previous_questions: list[str] | None = None

class QuestionResponse(BaseModel):
    questions: list[str] = Field(..., min_length=1)

    @field_validator("questions")
    @classmethod
    def validate_questions(cls, questions):
        if len(set(questions)) != len(questions):
            raise ValueError("중복 질문이 있습니다.")

        for q in questions:
            if len(q.strip()) < 10:
                raise ValueError("질문이 너무 짧습니다.")

            if len(q) > 200:
                raise ValueError("질문이 너무 깁니다.")

        return questions