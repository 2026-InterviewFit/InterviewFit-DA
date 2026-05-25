from pydantic import BaseModel

class QuestionRequest(BaseModel):
    count: int
    job: str
    resume_content: str | None = None

class QuestionResponse(BaseModel):
    questions: list[str]