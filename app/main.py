from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import settings
from app.VideoStorage.video_router import video_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="면접 영상 분석 AI 서버 (눈 랜드마크 / 깜빡임 / 꼬리질문)",
    version="0.1.0",
)

# React 프론트엔드 연결 대비 CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 기존 API Router
app.include_router(api_router)

# Video Router 추가
app.include_router(video_router)


@app.get("/")
def root():
    return {"message": "Interview-Fit AI Server is running"}

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Interview-Fit AI Server is running"}

@app.post("/analyze")
def analyze_video(data: dict):
    return {
        "message": "analysis complete",
        "result": {
            "gazeScore": 85,
            "blinkCount": 20,
            "postureScore": 90
        }
    }