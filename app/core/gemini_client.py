from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY가 .env 파일에 설정되지 않았습니다.")

# 도메인별 클라이언트 분리 (향후 도메인별 설정 분기 대비)
question_client = genai.Client(api_key=GEMINI_API_KEY)
followup_client = genai.Client(api_key=GEMINI_API_KEY)