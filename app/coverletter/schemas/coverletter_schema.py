from pydantic import BaseModel

class CoverLetterAnalysisRequest(BaseModel):
    coverletter: str

class CoverLetterQuestionItem(BaseModel):
    question_number: int
    question: str
    answer: str
    summary: str

class CoverLetterQuestionResponse(BaseModel):
    cover_letter_questions: list[CoverLetterQuestionItem]

class CoverLetterQuestionImportance(BaseModel):
    question_number: int
    importance_score: float

class CoverLetterImportanceResponse(BaseModel):
    question_importance_scores: list[CoverLetterQuestionImportance]

class StarItem(BaseModel):
    situation: str
    task: str
    action: str
    result: str

class CoverLetterQuestionAnalysisResult(BaseModel):
    star: StarItem
    strengths: list[str]
    improvements: list[str]
    skills: list[str]

class CoverLetterQuestionAnalysisItem(BaseModel):
    question_number: int
    question: str
    answer: str
    summary: str
    star: StarItem
    strengths: list[str]
    improvements: list[str]
    skills: list[str]
    importance_score: float | None

class JobRecommendation(BaseModel):
    job_name: str
    custom_job_name: str | None
    score: float
    job_reason: str

class CoverLetterDetailAnalysisResponse(BaseModel):
    questions_analysis: list[CoverLetterQuestionAnalysisItem]
    job: JobRecommendation