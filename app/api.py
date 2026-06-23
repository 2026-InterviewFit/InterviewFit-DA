# app/api.py
from fastapi import APIRouter

# 각 도메인 라우터 임포트
from app.speech_coaching.apis.speech_router import router as speech_router
from app.question.apis.question_router import router as question_router
from app.pose.apis.pose_router import router as pose_router

# eye-blink / followup 라우터 임포트
from app.eye.eye_blinking.apis.eye_router import router as eye_router
from app.followup.apis.followup_router import router as followup_router

# VideoStorage 라우터 임포트
from app.VideoStorage.video_router import video_router

# 공통 prefix /apis 설정
api_router = APIRouter(prefix="/apis")

# 음성 분석
api_router.include_router(
    speech_router,
    prefix="/speech",
    tags=["Speech"],
)

# 자세 분석
api_router.include_router(
    pose_router,
    prefix="/pose",
    tags=["Pose"],
)

# 질문 생성
api_router.include_router(
    question_router,
    prefix="/question",
    tags=["Question"],
)

# 눈 깜빡임 분석
api_router.include_router(
    eye_router,
    prefix="/eye-blinking",
    tags=["EyeBlinking"],
)

# 꼬리 질문 생성
api_router.include_router(
    followup_router,
    prefix="/followup",
    tags=["FollowUp"],
)

# 영상 저장/분석
api_router.include_router(
    video_router,
    prefix="/videos",
    tags=["Videos"],
)