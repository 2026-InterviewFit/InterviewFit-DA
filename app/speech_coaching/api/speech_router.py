from fastapi import APIRouter
from app.speech_coaching.services import speech_service

router = APIRouter()


# 개선된 API: source_type을 받아서 처리
@router.post("/analyze")
async def analyze_interview(
        source_type: str,  # "youtube" 또는 "minio"
        source_value: str,  # 유튜브 URL 또는 MinIO 객체 키(Path)
        job_category: str,
        interview_id: int
):
    # 서비스 계층에 출처 정보를 전달
    report = speech_service.process_interview_analysis(
        source_type, source_value, job_category, interview_id
    )

    if not report:
        return {"error": "분석 중 오류 발생"}

    return report