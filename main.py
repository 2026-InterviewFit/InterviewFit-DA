from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import api_router
from app.common.exceptions.custom_exception import CustomException
from app.common.exceptions.exception_handler import custom_exception_handler
from app.eye.eye_gaze.apis.gaze_router import router as gaze_router
from app.pt_interview.router.pt_report_router import (
    router as pt_report_router,
)
from app.pt_interview.router.pt_document_router import (
    router as pt_document_router,
)
from app.pt_interview.router.pt_question_router import (
    router as pt_question_router,
)


app = FastAPI(
    title="InterviewFit AI Server",
    description="InterviewFit AI analysis server",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 통합 라우터
app.include_router(api_router)

# 시선 분석
app.include_router(gaze_router)

# PT 면접
app.include_router(pt_document_router)
app.include_router(pt_question_router)
app.include_router(pt_report_router)

# 공통 예외 핸들러
app.add_exception_handler(
    CustomException,
    custom_exception_handler,
)


@app.get("/")
def root():
    return {
        "message": "InterviewFit AI Server is running",
        "docs": "/docs",
    }