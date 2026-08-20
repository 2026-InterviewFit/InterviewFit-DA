from pydantic import BaseModel, Field
from typing import Dict, List, Optional


# =========================================================
# 기존 전체 스피치 분석 요청
# =========================================================

class SpeechAnalyzeRequest(BaseModel):
    source_type: str
    source_value: str
    job_category: str
    interview_id: int
    interview_answer_id: Optional[int] = None
    question_id: Optional[int] = None
    original_question: Optional[str] = None


# =========================================================
# 꼬리질문용 STT 응답
# =========================================================

class SpeechTranscriptionResponse(BaseModel):
    """
    기본 질문/꼬리질문 답변 음성을
    Whisper로 텍스트 변환한 결과.

    꼬리질문 생성 단계에서는
    전체 스피치 분석이 필요하지 않으므로
    변환된 텍스트만 반환한다.
    """

    text: str


# =========================================================
# 기존 AI 평가 결과
# =========================================================

class AiEvaluation(BaseModel):
    star_score: Dict[str, int] = Field(
        ...,
        alias="star_score"
    )

    feedback: str

    keywords: List[str]

    model_answer: str = Field(
        ...,
        alias="model_answer"
    )

    class Config:
        populate_by_name = True


# =========================================================
# 필러워드 분석
# =========================================================

class FillerWordAnalysis(BaseModel):
    details: Dict[str, int]

    total_count: int = Field(
        ...,
        alias="total_count"
    )

    class Config:
        populate_by_name = True


# =========================================================
# 반복 표현
# =========================================================

class RepeatedExpression(BaseModel):
    expression: str
    count: int


# =========================================================
# 말하기 속도
# =========================================================

class SpeechPaceAnalysis(BaseModel):
    evaluation: str

    feedback: str

    characters_per_second: float = Field(
        ...,
        alias="characters_per_second"
    )

    class Config:
        populate_by_name = True


# =========================================================
# 침묵 분석
# =========================================================

class SilenceAnalysis(BaseModel):
    total_silence_seconds: float = Field(
        ...,
        alias="total_silence_seconds"
    )

    pause_rate_percent: float = Field(
        ...,
        alias="pause_rate_percent"
    )

    critical_silence_count: int = Field(
        ...,
        alias="critical_silence_count"
    )


# =========================================================
# 기존 전체 스피치 분석 응답
# =========================================================

class SpeechReportResponse(BaseModel):
    interview_id: int

    interview_answer_id: Optional[int] = None

    question_id: Optional[int] = None

    original_question: Optional[str] = None

    job_category: str

    text_script: str

    filler_word_analysis: FillerWordAnalysis

    repeated_expressions: List[
        RepeatedExpression
    ]

    speech_pace_analysis: SpeechPaceAnalysis

    silence_analysis: SilenceAnalysis

    ai_evaluation: AiEvaluation