import io
import os
from typing import List

from dotenv import load_dotenv
from minio import Minio
from pptx import Presentation
from pypdf import PdfReader

from app.pt_interview.schema.pt_document_schema import (
    PtDocumentAnalyzeResponse,
    PtSlideAnalyzeResult,
)
from app.pt_interview.service.pt_summary_service import (
    PtSummaryService,
)

load_dotenv()


class PtDocumentService:

    def __init__(self):

        self.minio_client = Minio(
            os.getenv(
                "MINIO_ENDPOINT",
                "localhost:9000",
            ),
            access_key=os.getenv(
                "MINIO_ACCESS_KEY"
            ),
            secret_key=os.getenv(
                "MINIO_SECRET_KEY"
            ),
            secure=os.getenv(
                "MINIO_SECURE",
                "False",
            ).lower() == "true",
        )

        self.bucket_name = os.getenv(
            "MINIO_BUCKET",
            "videos",
        )

        self.summary_service = PtSummaryService()

    def analyze_document(
        self,
        object_key: str,
    ) -> PtDocumentAnalyzeResponse:

        """
        PT 발표자료 분석

        1. MinIO 파일 다운로드
        2. 파일 형식 확인
        3. PPTX / PDF 텍스트 추출
        4. 슬라이드별 Gemini 요약
        5. 전체 슬라이드 텍스트 결합
        6. Gemini 전체 발표자료 요약
        7. 분석 결과 반환
        """

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

        # 슬라이드별 Gemini 요약
        slides = self._summarize_slides(
            slides
        )

        # 전체 발표자료 텍스트 생성
        document_text = self._build_document_text(
            slides
        )

        # Gemini 전체 발표자료 요약
        document_summary = (
            self.summary_service
            .summarize_document(
                document_text
            )
        )

        return PtDocumentAnalyzeResponse(
            object_key=object_key,
            file_type=extension.replace(
                ".",
                "",
            ),
            total_slides=len(slides),
            summary=document_summary,
            slides=slides,
        )

    def _download_from_minio(
        self,
        object_key: str,
    ) -> bytes:

        response = None

        try:

            response = (
                self.minio_client
                .get_object(
                    self.bucket_name,
                    object_key,
                )
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

        return os.path.splitext(
            object_key
        )[1].lower()

    def _extract_pdf(
        self,
        file_bytes: bytes,
    ) -> List[PtSlideAnalyzeResult]:

        reader = PdfReader(
            io.BytesIO(
                file_bytes
            )
        )

        slides = []

        for index, page in enumerate(
            reader.pages,
            start=1,
        ):

            text = (
                page.extract_text()
                or ""
            )

            text = text.strip()

            slides.append(
                PtSlideAnalyzeResult(
                    slide_number=index,
                    title=self._extract_title(
                        text
                    ),
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
            io.BytesIO(
                file_bytes
            )
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

                    text_parts.append(
                        text
                    )

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

    def _summarize_slides(
        self,
        slides: List[
            PtSlideAnalyzeResult
        ],
    ) -> List[
        PtSlideAnalyzeResult
    ]:

        """
        슬라이드별 Gemini 요약 생성
        """

        summarized_slides = []

        for slide in slides:

            summary = (
                self.summary_service
                .summarize_slide(
                    slide_number=slide.slide_number,
                    title=slide.title,
                    extracted_text=slide.extracted_text,
                )
            )

            summarized_slides.append(
                PtSlideAnalyzeResult(
                    slide_number=slide.slide_number,
                    title=slide.title,
                    extracted_text=slide.extracted_text,
                    summary=summary,
                )
            )

        return summarized_slides

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

    def _build_document_text(
        self,
        slides: List[
            PtSlideAnalyzeResult
        ],
    ) -> str:

        """
        Gemini 전체 발표자료 요약에 사용할
        슬라이드 텍스트 생성
        """

        text_parts = []

        for slide in slides:

            if not slide.extracted_text:
                continue

            text_parts.append(
                f"""
[슬라이드 {slide.slide_number}]
제목: {slide.title or "없음"}

원본 내용:
{slide.extracted_text}

슬라이드 요약:
{slide.summary or "없음"}
""".strip()
            )

        return "\n\n".join(
            text_parts
        )