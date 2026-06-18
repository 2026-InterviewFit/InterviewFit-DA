from fastapi import APIRouter
from app.speech_coaching.services import speech_service
from app.speech_coaching.schemas.speech_schema import SpeechAnalyzeRequest

router = APIRouter()

@router.post("/analyze")
async def analyze_interview(request: SpeechAnalyzeRequest):
    return speech_service.process_interview_analysis(
        source_type=request.source_type,
        source_value=request.source_value,
        job_category=request.job_category,
        interview_id=request.interview_id
    )