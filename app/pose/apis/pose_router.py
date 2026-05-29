import cv2

from fastapi import APIRouter

from app.common.exceptions.custom_exception import CustomException
from app.common.exceptions.error_code import ErrorCode
from app.common.exceptions.status_code import StatusCode
from app.pose.schemas.pose_schema import PoseAnalysisResponse, PoseAnalysisRequest
from app.pose.services.pose_service import process_video

router = APIRouter(prefix="/pose",tags=["Pose"])

@router.post("/analyze", response_model=list[PoseAnalysisResponse])
async def analyze_pose(request: PoseAnalysisRequest):
    # 프레임 읽어오는 함수를 사용할 예정 (아래 코드 수정 필요)
    # video_url로 프레임 추출 (이미 있다고 가정)
    # frames = extract_frames(request.video_object_key)
    image_paths = ["./images/online_interview.jpeg",
                   "./images/online_interview_2.jpeg",
                   "./images/online_interview_6.png",
                   "./images/online_interview_6.png",
                   "./images/online_interview_3.jpeg",
                   "./images/online_interview_4.jpeg",
                   "./images/online_interview_5.jpeg"]

    frames = []
    for path in image_paths:
        frame = cv2.imread(path)
        frames.append(frame)

    if len(frames) == 0:
        raise CustomException(
            status=StatusCode.BAD_REQUEST,
            code=ErrorCode.FRAME_NOT_FOUND,
            message="영상에서 프레임을 추출할 수 없습니다."
        )

    results = process_video(frames, fps=1) # fps: 1초당 1프레임 (나중에 수정 필요)

    print(results)
    return results