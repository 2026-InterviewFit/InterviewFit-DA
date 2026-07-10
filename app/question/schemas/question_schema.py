from pydantic import BaseModel, Field, field_validator

class QuestionRequest(BaseModel):
    count: int
    job: str
    cover_letter: str | None = None

class QuestionResponse(BaseModel):
    questions: list[str]

class QuestionValidation(BaseModel):
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