import cv2

from fastapi import APIRouter

from app.pose.schemas.pose_schema import PoseAnalysisResponse, PoseAnalysisRequest
from app.pose.services.pose_service import process_video

router = APIRouter(prefix="/api/pose",tags=["Pose"])

@router.post("/analyze", response_model=list[PoseAnalysisResponse])
async def analyze_pose(request: PoseAnalysisRequest):
    print(request.video_url)

    # 프레임 읽어오는 함수를 사용할 예정 (아래 코드 수정 필요)
    image_paths = ["./images/online_interview.jpeg",
                   # "./images/online_interview_2.jpeg",
                   "./images/online_interview_3.jpeg",
                   "./images/online_interview_4.jpeg",
                   "./images/online_interview_5.jpeg"]
                   # "./images/online_interview_6.png"]

    frames = []
    for path in image_paths:
        frame = cv2.imread(path)
        if frame is None:
            raise Exception(f"이미지 로드 실패: {path}")
        frames.append(frame)

    # video_url로 프레임 추출 (이미 있다고 가정)
    # frames = extract_frames(request.video_url)

    results = process_video(frames, fps=1) # fps: 1초당 1프레임

    return results

    # result = process_posture(
    #     frame=frame,
    #     timestamp=0
    # )
    #
    # return PoseAnalysisResponse(
    #     timestamp=result["timestamp"],
    #     shoulder_tilt=result["shoulder_tilt"],
    #     face_direction=result["face_direction"],
    #     shoulder_status=result["shoulder_status"],
    #     face_status=result["face_status"]
    # )