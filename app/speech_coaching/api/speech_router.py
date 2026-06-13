from fastapi import APIRouter
from app.speech_coaching.services import speech_service
from fastapi import APIRouter, HTTPException
router = APIRouter()


# 개선된 API: source_type을 받아서 처리
@router.post("/analyze")
async def analyze_interview(
        source_type: str,  # "youtube" 또는 "minio", 또는 "local"
        source_value: str,  # 유튜브 URL 또는 MinIO 객체 키(Path), 파일 경로
        job_category: str,
        interview_id: int
):
    # 서비스 계층에 출처 정보를 전달
    report = speech_service.process_interview_analysis(
        source_type, source_value, job_category, interview_id
    )

    # 서비스에서 에러 메시지를 반환했는지 확인 (딕셔너리에 "error" 키가 있는지 확인)
    if isinstance(report, dict) and "error" in report:
        raise HTTPException(status_code=400, detail=report["error"])

    if not report:
        raise HTTPException(status_code=500, detail="분석 중 예기치 않은 오류가 발생했습니다.")

    return report