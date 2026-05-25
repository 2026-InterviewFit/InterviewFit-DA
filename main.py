from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.eye_blinking.api.v1.router import api_router
from app.eye_blinking.core.config import settings

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

app.include_router(api_router)


@app.get("/")
def root():
    return {"message": "Interview-Fit AI Server is running"}


@app.get("/health")
def health():
    return {"status": "ok"}