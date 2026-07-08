from pydantic import BaseModel
from typing import Dict, List, Optional

# FastAPI 요청용 모델
class SpeechAnalyzeRequest(BaseModel):
    source_type: str
    source_value: str
    job_category: str
    interview_id: int

# AI 평가 결과용 세부 모델
class AiEvaluation(BaseModel):
    star_score: Dict[str, int]
    feedback: str
    keywords: List[str]
    model_answer: str

# 필러워드(습관어) 분석 모델
class FillerWordAnalysis(BaseModel):
    details: Dict[str, int]
    total_count: int

# 반복 표현 단건 모델
class RepeatedExpression(BaseModel):
    expression: str
    count: int

# 말하기 속도 분석 모델
class SpeechPaceAnalysis(BaseModel):
    evaluation: str
    feedback: str
    characters_per_second: float

# 음량(데시벨) 및 음량 안정성 모델
class VolumeAnalysis(BaseModel):
    volume_db_mean: float
    volume_db_std: float

# 침묵 구간 및 침묵 비율 모델
class SilenceAnalysis(BaseModel):
    total_silence_seconds: float
    pause_rate_percent: float
    critical_silence_count: int

# 최종 응답용 모델
class SpeechReportResponse(BaseModel):
    text_script: str
    filler_word_analysis: FillerWordAnalysis
    repeated_expressions: List[RepeatedExpression]
    speech_pace_analysis: SpeechPaceAnalysis
    volume_analysis: VolumeAnalysis
    silence_analysis: SilenceAnalysis
    ai_evaluation: AiEvaluation
    model_answer: str
    job_category: str
    interview_id: int