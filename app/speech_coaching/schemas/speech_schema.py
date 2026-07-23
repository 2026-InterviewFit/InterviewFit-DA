from pydantic import BaseModel, Field
from typing import Dict, List, Optional


# FastAPI 요청 모델
class SpeechAnalyzeRequest(BaseModel):
    source_type: str
    source_value: str
    job_category: str
    interview_id: int
    interview_answer_id: Optional[int] = None
    question_id: Optional[int] = None
    original_question: Optional[str] = None


# AI 평가 결과용 세부 모델
class AiEvaluation(BaseModel):
    star_score: Dict[str, int] = Field(..., alias="star_score")
    feedback: str
    keywords: List[str]
    model_answer: str = Field(..., alias="model_answer")

    class Config:
        populate_by_name = True


# 필러워드(습관어) 분석 모델
class FillerWordAnalysis(BaseModel):
    details: Dict[str, int]
    total_count: int = Field(..., alias="total_count")

    class Config:
        populate_by_name = True


# 반복 표현 모델
class RepeatedExpression(BaseModel):
    expression: str
    count: int


# 말하기 속도 분석 모델
class SpeechPaceAnalysis(BaseModel):
    evaluation: str
    feedback: str
    characters_per_second: float = Field(..., alias="characters_per_second")

    class Config:
        populate_by_name = True


# 침묵 분석 모델
class SilenceAnalysis(BaseModel):
    total_silence_seconds: float = Field(..., alias="total_silence_seconds")
    pause_rate_percent: float = Field(..., alias="pause_rate_percent")
    critical_silence_count: int = Field(..., alias="critical_silence_count")


# 최종 응답 모델
class SpeechReportResponse(BaseModel):
    interview_id: int
    interview_answer_id: Optional[int] = None
    question_id: Optional[int] = None
    original_question: Optional[str] = None
    job_category: str

    text_script: str

    #total_filler_count: int
    filler_word_analysis: FillerWordAnalysis
    repeated_expressions: List[RepeatedExpression]

    speech_pace_analysis: SpeechPaceAnalysis
    silence_analysis: SilenceAnalysis
    ai_evaluation: AiEvaluation
    #model_answer: str