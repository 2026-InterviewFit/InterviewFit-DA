from fastapi import APIRouter, UploadFile, File

from app.ocr.schemas.ocr_schema import OcrResponse
from app.ocr.services.ocr_service import extract_text

router = APIRouter()

@router.post("/extract", response_model=OcrResponse)
async def analyze_pose(file: UploadFile = File(...)):
    text = await extract_text(file)
    return OcrResponse(text=text)