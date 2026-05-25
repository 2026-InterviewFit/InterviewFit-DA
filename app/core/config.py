from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Interview-Fit AI Server"

    # CORS 허용 origin (운영 환경에서는 명시적 도메인으로 변경 권장)
    BACKEND_CORS_ORIGINS: list[str] = ["*"]

    # Gemini API 키 (꼬리 질문 생성용, 추후 사용)
    GEMINI_API_KEY: str = ""

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()