from fastapi import APIRouter, HTTPException

from app.pt_interview.schema.pt_report_schema import (
    PtReportGenerateRequest,
    PtReportGenerateResponse,
)
from app.pt_interview.service.pt_report_service import PtReportService


router = APIRouter(
    prefix="/apis/pt-report",
    tags=["PT Report"],
)

pt_report_service = PtReportService()


@router.post(
    "/generate",
    response_model=PtReportGenerateResponse,
)
def generate_pt_report(
    request: PtReportGenerateRequest,
) -> PtReportGenerateResponse:
    try:
        return pt_report_service.generate_report(request)

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"PT 리포트 생성 중 오류가 발생했습니다: {error}",
        ) from error