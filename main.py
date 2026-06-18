# main.py
from fastapi import FastAPI
from app.api import api_router  # 통합 라우터 불러오기
from app.common.exceptions.custom_exception import CustomException
from app.common.exceptions.exception_handler import custom_exception_handler

app = FastAPI(title="InterviewFit AI Server")

# 통합 라우터 연결
app.include_router(api_router)
app.add_exception_handler(CustomException, custom_exception_handler)

@app.get("/")
def root():
    return {"message": "InterviewFit AI Server is running"}