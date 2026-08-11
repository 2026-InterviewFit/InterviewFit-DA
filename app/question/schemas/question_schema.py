from pydantic import BaseModel, Field, field_validator

class StarItem(BaseModel):
    situation: str
    task: str
    action: str
    result: str

class CoverLetterQuestionAnalysisItem(BaseModel):
    question_number: int
    summary: str
    star: StarItem = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    importance_score: float | None = None

class QuestionRequest(BaseModel):
    count: int
    job: str
    analyses: list[CoverLetterQuestionAnalysisItem] = Field(default_factory=list)
    previous_questions: list[str] | None = None
    reusable_questions: list[str] | None = None

class QuestionItem(BaseModel):
    question: str
    question_number: int = Field(..., ge=1) # 자소서 문항 번호

class QuestionResponse(BaseModel):
    questions: list[QuestionItem] = Field(..., min_length=1)

    @field_validator("questions")
    @classmethod
    def validate_questions(cls, questions):
        question_texts = [item.question for item in questions]

        # 질문 중복 검사
        if len(set(question_texts)) != len(question_texts):
            raise ValueError("중복 질문이 있습니다.")

        # 질문 길이 검사
        for item in questions:
            question = item.question.strip()

            if len(question) < 10:
                raise ValueError("질문이 너무 짧습니다.")

            if len(question) > 200:
                raise ValueError("질문이 너무 깁니다.")

        return questions