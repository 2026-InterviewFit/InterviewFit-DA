from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Interview-Fit AI Server"

    # CORS 허용 origin. 개발 단계에서는 전체 허용, 운영에서는 FE 도메인으로 제한 권장.
    BACKEND_CORS_ORIGINS: list[str] = ["*"]

    # Gemini API 설정. 키가 비어 있으면 question/followup 서비스가 fallback 응답을 사용한다.
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash-lite"

    # 팀원 VideoStorage 코드와 eye-blink가 함께 사용할 MinIO 설정.
    # Local: localhost:9000 / Docker: minio:9000
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin123"
    MINIO_BUCKET: str = "videos"
    MINIO_SECURE: bool = False

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()


# class Settings(BaseSettings):
#     PROJECT_NAME: str = "Interview-Fit AI Server"
#
#     # CORS 허용 origin (운영 환경에서는 명시적 도메인으로 변경 권장)
#     BACKEND_CORS_ORIGINS: list[str] = ["*"]
#
#     # Gemini API 키 (꼬리 질문 생성용, 추후 사용)
#     GEMINI_API_KEY: str = ""
#
#     class Config:
#         env_file = ".env"
#         case_sensitive = True
#
#
# settings = Settings()