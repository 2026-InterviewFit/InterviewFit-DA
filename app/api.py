# app/api.py
from fastapi import APIRouter

# 각 도메인 라우터 임포트
from app.speech_coaching.apis.speech_router import router as speech_router
from app.question.apis.question_router import router as question_router
from app.pose.apis.pose_router import router as pose_router

# 공통 prefix /api 설정
api_router = APIRouter(prefix="/apis")

api_router.include_router(speech_router, prefix="/speech", tags=["Speech"])

api_router.include_router(pose_router, prefix="/pose", tags=["Pose"])
api_router.include_router(question_router, prefix="/question", tags=["Question"])
