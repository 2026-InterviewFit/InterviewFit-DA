import cv2

from fastapi import APIRouter

from app.VideoStorage.minio_downloader import download_video_from_minio
from app.common.exceptions.custom_exception import CustomException
from app.common.exceptions.error_code import ErrorCode
from app.common.exceptions.status_code import StatusCode
from app.frame.service.frame_service import analyze_frames
from app.pose.schemas.pose_schema import PoseAnalysisResponse, PoseAnalysisRequest
from app.pose.services.pose_service import process_video

router = APIRouter(prefix="/pose",tags=["Pose"])

@router.post("/analyze", response_model=list[PoseAnalysisResponse])
async def analyze_pose(request: PoseAnalysisRequest):
    local_video_path = download_video_from_minio(request.video_object_key)
    frame_result = analyze_frames(local_video_path)

    frames = []

    for path in frame_result["framePaths"]:
        frame = cv2.imread(path)

        if frame is not None:
            frames.append(frame)

    if len(frames) == 0:
        raise CustomException(
            status=StatusCode.BAD_REQUEST,
            code=ErrorCode.FRAME_NOT_FOUND,
            message="영상에서 프레임을 추출할 수 없습니다."
        )

    results = process_video(frames, fps=1)

    return results