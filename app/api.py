# app/api.py
from fastapi import APIRouter

# 각 도메인 라우터 임포트
from app.speech_coaching.apis.speech_router import router as speech_router

# 공통 prefix /api 설정
api_router = APIRouter(prefix="/apis")

api_router.include_router(speech_router, prefix="/speech", tags=["Speech"])
