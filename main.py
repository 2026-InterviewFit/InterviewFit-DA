from fastapi import FastAPI
from app.speech_coaching.api import speech_router
import uvicorn

app = FastAPI()

# 라우터 연결
app.include_router(speech_router.router, prefix="/speech")

@app.get("/")
def read_root():
    return {"message": "AI 스피치 코칭 서버 실행 중"}

if __name__ == "__main__":
    # 이 코드가 있어야 '파이썬 실행'을 했을 때 서버가 켜지고 종료되지 않습니다.
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)