from fastapi import APIRouter, HTTPException

from app.pt_interview.schema.pt_document_schema import (
    PtDocumentAnalyzeRequest,
    PtDocumentAnalyzeResponse,
)
from app.pt_interview.service.pt_document_service import (
    PtDocumentService,
)


router = APIRouter(
    prefix="/apis/pt-document",
    tags=["PT Document"],
)

service = PtDocumentService()


@router.post(
    "/analyze",
    response_model=PtDocumentAnalyzeResponse,
)
def analyze_pt_document(
    request: PtDocumentAnalyzeRequest,
):
    try:

        return service.analyze_document(
            request.object_key
        )

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )