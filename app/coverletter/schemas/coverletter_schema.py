from pydantic import BaseModel

class CoverLetterAnalysisRequest(BaseModel):
    coverletter: str

class StarDto(BaseModel):
    situation: str
    task: str
    action: str
    result: str

class CoverLetterQuestionAnalysisDto(BaseModel):
    question_number: int
    question: str
    answer: str
    summary: str
    star: StarDto
    strengths: list[str]
    improvements: list[str]
    skills: list[str]
    keywords: list[str]
    importance_score: float

class CoverLetterAnalysisResponse(BaseModel):
    questions_analysis: list[CoverLetterQuestionAnalysisDto]
    job_name: str
    custom_job_name: str | None
    score: float
    job_reason: str