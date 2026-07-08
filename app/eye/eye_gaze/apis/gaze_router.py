from fastapi import APIRouter, HTTPException

from app.eye.eye_gaze.schemas.gaze_schema import (
    GazeAnalyzeRequest,
    GazeAnalyzeResponse,
)
from app.eye.eye_gaze.services.gaze_service import GazeService


router = APIRouter(
    prefix="/eye-gaze",
    tags=["Eye Gaze"],
)

gaze_service = GazeService()


@router.post("/analyze", response_model=GazeAnalyzeResponse)
def analyze_gaze(request: GazeAnalyzeRequest):
    try:
        return gaze_service.analyze_video(request.video_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))