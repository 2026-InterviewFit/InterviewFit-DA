from pydantic import BaseModel

class QuestionRequest(BaseModel):
    count: int
    job: str
    cover_letter: str | None = None

class QuestionResponse(BaseModel):
    questions: list[str]