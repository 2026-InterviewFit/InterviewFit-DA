from google import genai

from app.core.config import settings


def create_gemini_client():
    """
    GEMINI_API_KEY가 없을 때도 FastAPI 서버가 죽지 않도록
    import 시점에 예외를 발생시키지 않는다.
    """
    if not settings.GEMINI_API_KEY:
        return None

    return genai.Client(api_key=settings.GEMINI_API_KEY)


# 도메인별 클라이언트 분리
question_client = create_gemini_client()
followup_client = create_gemini_client()

# load_dotenv()
#
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
#
# if not GEMINI_API_KEY:
#     raise ValueError("GEMINI_API_KEY가 .env 파일에 설정되지 않았습니다.")
#
# # 도메인별 클라이언트 분리 (향후 도메인별 설정 분기 대비)
# question_client = genai.Client(api_key=GEMINI_API_KEY)
# followup_client = genai.Client(api_key=GEMINI_API_KEY)