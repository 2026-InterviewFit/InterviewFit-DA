from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import api_router
from app.common.exceptions.custom_exception import CustomException
from app.common.exceptions.exception_handler import custom_exception_handler
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="면접 영상 분석 AI 서버",
    version="0.1.0",
)

app.add_exception_handler(CustomException, custom_exception_handler)

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