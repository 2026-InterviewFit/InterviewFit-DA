from fastapi import APIRouter

from app.coverletter.schemas.coverletter_schema import CoverLetterAnalysisResponse, CoverLetterAnalysisRequest
from app.coverletter.services.coverletter_service import generate_cover_letter_analysis

router = APIRouter()

@router.post("/analysis", response_model=CoverLetterAnalysisResponse)
async def analyze_cover_letter(request: CoverLetterAnalysisRequest):
    return await generate_cover_letter_analysis(coverletter=request.coverletter)