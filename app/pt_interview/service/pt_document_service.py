import io
import os
from typing import List

from minio import Minio
from pptx import Presentation
from pypdf import PdfReader

from app.pt_interview.schema.pt_document_schema import (
    PtDocumentAnalyzeResponse,
    PtSlideAnalyzeResult,
)
from dotenv import load_dotenv

load_dotenv()


class PtDocumentService:

    def __init__(self):
        self.minio_client = Minio(
            os.getenv("MINIO_ENDPOINT", "localhost:9000"),
            access_key=os.getenv("MINIO_ACCESS_KEY"),
            secret_key=os.getenv("MINIO_SECRET_KEY"),
            secure=os.getenv("MINIO_SECURE", "False").lower() == "true",
        )

        self.bucket_name = os.getenv("MINIO_BUCKET", "videos")

    def analyze_document(
        self,
        object_key: str,
    ) -> PtDocumentAnalyzeResponse:

        file_bytes = self._download_from_minio(
            object_key
        )

        extension = self._get_extension(
            object_key
        )

        if extension == ".pdf":
            slides = self._extract_pdf(
                file_bytes
            )

        elif extension == ".pptx":
            slides = self._extract_pptx(
                file_bytes
            )

        else:
            raise ValueError(
                "지원하지 않는 파일 형식입니다. "
                "PDF 또는 PPTX 파일만 분석할 수 있습니다."
            )

        return PtDocumentAnalyzeResponse(
            object_key=object_key,
            file_type=extension.replace(".", ""),
            total_slides=len(slides),
            slides=slides,
        )

    def _download_from_minio(
        self,
        object_key: str,
    ) -> bytes:

        response = None

        try:
            response = self.minio_client.get_object(
                self.bucket_name,
                object_key,
            )

            return response.read()

        except Exception as e:
            raise RuntimeError(
                f"MinIO 파일 다운로드 실패: {e}"
            ) from e

        finally:
            if response is not None:
                response.close()
                response.release_conn()

    def _get_extension(
        self,
        object_key: str,
    ) -> str:

        extension = os.path.splitext(
            object_key
        )[1].lower()

        return extension

    def _extract_pdf(
        self,
        file_bytes: bytes,
    ) -> List[PtSlideAnalyzeResult]:

        reader = PdfReader(
            io.BytesIO(file_bytes)
        )

        slides = []

        for index, page in enumerate(
            reader.pages,
            start=1,
        ):
            text = page.extract_text() or ""

            text = text.strip()

            slides.append(
                PtSlideAnalyzeResult(
                    slide_number=index,
                    title=self._extract_title(text),
                    extracted_text=text,
                    summary=None,
                )
            )

        return slides

    def _extract_pptx(
        self,
        file_bytes: bytes,
    ) -> List[PtSlideAnalyzeResult]:

        presentation = Presentation(
            io.BytesIO(file_bytes)
        )

        slides = []

        for index, slide in enumerate(
            presentation.slides,
            start=1,
        ):

            text_parts = []

            for shape in slide.shapes:

                if not hasattr(
                    shape,
                    "text",
                ):
                    continue

                text = shape.text.strip()

                if text:
                    text_parts.append(text)

            extracted_text = "\n".join(
                text_parts
            )

            slides.append(
                PtSlideAnalyzeResult(
                    slide_number=index,
                    title=self._extract_title(
                        extracted_text
                    ),
                    extracted_text=extracted_text,
                    summary=None,
                )
            )

        return slides

    def _extract_title(
        self,
        text: str,
    ) -> str | None:

        if not text:
            return None

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        if not lines:
            return None

        return lines[0][:200]