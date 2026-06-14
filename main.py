from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.common.exceptions.custom_exception import CustomException
from app.common.exceptions.exception_handler import custom_exception_handler
from app.core.config import settings

from app.eye_blinking.api.v1.eye_router import api_router as eye_blinking_router
from app.followup.apis.followup_router import router as followup_router
from app.pose.apis.pose_router import router as pose_router
from app.question.apis.question_router import router as question_router
from app.VideoStorage.video_router import video_router

# speech 라우터 임시 비활성화
from app.speech_coaching.api.speech_router import router as speech_router
from dotenv import load_dotenv
import os
load_dotenv()

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

app.include_router(eye_blinking_router)
app.include_router(followup_router)
app.include_router(pose_router)
app.include_router(question_router)
app.include_router(video_router)

# speech 기능 임시 비활성화
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