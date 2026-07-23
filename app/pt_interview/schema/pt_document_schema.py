from typing import List, Optional

from pydantic import BaseModel, Field


class PtDocumentAnalyzeRequest(BaseModel):
    object_key: str = Field(
        ...,
        description="MinIO에 저장된 발표자료 object key",
    )


class PtSlideAnalyzeResult(BaseModel):
    slide_number: int

    title: Optional[str] = None

    extracted_text: str = ""

    summary: Optional[str] = None


class PtDocumentAnalyzeResponse(BaseModel):
    object_key: str

    file_type: str

    total_slides: int

    summary: Optional[str] = None

    slides: List[PtSlideAnalyzeResult]