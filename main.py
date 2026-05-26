from fastapi import FastAPI
from app.speech_coaching.api import speech_router

app = FastAPI()

# 라우터 연결
app.include_router(speech_router.router, prefix="/speech")

@app.get("/")
def read_root():
    return {"message": "AI 스피치 코칭 서버 실행 중"}