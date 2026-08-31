from fastapi import APIRouter
from app.pose.schemas.pose_schema import PoseAnalysisResponse, PoseAnalysisRequest
from app.pose.services.pose_service import analyze_pose_video

router = APIRouter()

@router.post("/analyze", response_model=list[PoseAnalysisResponse])
def analyze_pose(request: PoseAnalysisRequest):
    return analyze_pose_video(video_object_key=request.video_object_key)