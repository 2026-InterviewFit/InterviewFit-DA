from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.common.exceptions.custom_exception import CustomException
from app.common.exceptions.exception_handler import custom_exception_handler
from app.core.config import settings

# eye-blinking 라우터
# develop 브랜치 기준 실제 파일명이 router.py가 아니라 eye_router.py라서
# app.eye_blinking.api.v1.eye_router 로 import 해야 함
from app.eye_blinking.api.v1.eye_router import api_router as eye_blinking_router

# followup 라우터
from app.followup.apis.followup_router import router as followup_router

# pose 라우터
from app.pose.apis.pose_router import router as pose_router

# video storage 라우터
from app.VideoStorage.video_router import video_router

# speech 라우터
from app.speech_coaching.api.speech_router import router as speech_router


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="면접 영상 분석 AI 서버",
    version="0.1.0",
)

app.add_exception_handler(
    CustomException,
    custom_exception_handler,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 각 기능 라우터 등록
app.include_router(eye_blinking_router)
app.include_router(followup_router)
app.include_router(pose_router)
app.include_router(video_router)

# speech_router는 기존 코드 기준 prefix="/speech"로 등록
app.include_router(speech_router, prefix="/speech")


@app.get("/")
def root():
    return {"message": "Interview-Fit AI Server is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )