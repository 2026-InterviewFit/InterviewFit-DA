from pydantic import BaseModel, Field


class PtQuestionAnswerRequest(BaseModel):
    """
    질문 1개와 사용자의 실제 답변.
    """
    question_number: int
    question: str
    answer: str = ""


class PtQuestionAnalysis(BaseModel):
    """
    질문별 상세 평가 결과.
    """
    question_number: int
    question: str
    answer: str

    understanding_score: int = Field(
        ge=0,
        le=100,
    )

    logic_score: int = Field(
        ge=0,
        le=100,
    )

    specificity_score: int = Field(
        ge=0,
        le=100,
    )

    evidence_score: int = Field(
        ge=0,
        le=100,
    )

    overall_score: int = Field(
        ge=0,
        le=100,
    )

    strengths: str
    weaknesses: str

    """
    사용자가 실제 면접에서
    어떤 식으로 답하면 더 좋은지 설명.
    """
    improvement: str

    """
    개선된 답변 예시.
    """
    improved_answer: str


class PtReportGenerateRequest(BaseModel):

    """
    PPT/PDF 자체를 분석한 요약.
    """
    presentation_material_summary: str = ""

    """
    실제 사용자가 발표하면서 말한 내용의 STT.
    """
    presentation_transcript: str = ""

    """
    질문 + 실제 답변.
    """
    qa_items: list[PtQuestionAnswerRequest] = Field(
        default_factory=list
    )

    eye_scores: list[float] = Field(
        default_factory=list
    )

    blink_scores: list[float] = Field(
        default_factory=list
    )

    pose_scores: list[float] = Field(
        default_factory=list
    )

    speech_scores: list[float] = Field(
        default_factory=list
    )


class PtReportGenerateResponse(BaseModel):

    content_score: int = Field(
        ge=0,
        le=100,
    )

    structure_score: int = Field(
        ge=0,
        le=100,
    )

    delivery_score: int = Field(
        ge=0,
        le=100,
    )

    qa_score: int = Field(
        ge=0,
        le=100,
    )

    overall_score: int = Field(
        ge=0,
        le=100,
    )

    """
    업로드한 PPT/PDF의 핵심 내용.
    """
    presentation_material_summary: str

    """
    사용자가 실제로 발표한 내용의 요약.
    """
    actual_presentation_summary: str

    """
    발표자료와 실제 발표를 비교한 분석.
    예:
    - 어떤 내용을 잘 설명했는지
    - 어떤 내용을 빠뜨렸는지
    - 자료만 읽은 부분이 있는지
    """
    presentation_comparison: str

    """
    질문 1~5 각각에 대한 상세 분석.
    """
    question_analyses: list[PtQuestionAnalysis] = Field(
        default_factory=list
    )

    """
    전체 PT 면접 종합 피드백.
    """
    strengths: str

    weaknesses: str

    improvement: str