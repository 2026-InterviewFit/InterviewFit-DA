from app.core.config import settings
from app.core.gemini_client import pt_interview_client
from app.pt_interview.prompts.pt_document_prompts import (
    build_pt_document_summary_prompt,
)


class PtSummaryService:

    """
    PT 발표자료 Gemini 요약 서비스

    기능:
    1. 슬라이드별 내용 요약
    2. 발표자료 전체 내용 요약
    """

    def summarize_document(
        self,
        slides_text: str,
    ) -> str | None:

        """
        발표자료 전체 내용을 종합해서 요약한다.
        """

        if not slides_text.strip():
            return None

        # Gemini API Key가 없는 경우
        if pt_interview_client is None:
            return self._fallback_summary(
                slides_text
            )

        try:

            prompt = (
                build_pt_document_summary_prompt(
                    slides_text
                )
            )

            response = (
                pt_interview_client
                .models
                .generate_content(
                    model=getattr(
                        settings,
                        "GEMINI_MODEL",
                        "gemini-3.5-flash-lite",
                    ),
                    contents=prompt,
                )
            )

            if (
                response is None
                or not response.text
            ):
                return self._fallback_summary(
                    slides_text
                )

            return response.text.strip()

        except Exception as e:

            print(
                f"Gemini PT document summary error: {e}"
            )

            return self._fallback_summary(
                slides_text
            )

    def summarize_slide(
        self,
        slide_number: int,
        title: str | None,
        extracted_text: str,
    ) -> str | None:

        """
        하나의 슬라이드 내용을 분석해서
        핵심 내용을 요약한다.
        """

        if not extracted_text.strip():
            return None

        # Gemini API Key가 없는 경우
        if pt_interview_client is None:
            return self._fallback_slide_summary(
                extracted_text
            )

        prompt = f"""
당신은 PT 면접 발표자료 분석 전문가입니다.

아래 PT 발표자료의 한 슬라이드를 분석하여
면접 준비에 필요한 핵심 내용을 요약하세요.

[슬라이드 번호]
{slide_number}

[슬라이드 제목]
{title or "없음"}

[슬라이드 내용]
{extracted_text}

[요약 기준]

1. 슬라이드에서 전달하려는 핵심 내용을 파악합니다.
2. 중요한 주장, 근거, 수치, 기술, 사례가 있다면 포함합니다.
3. 단순히 원문을 반복하지 말고 의미 중심으로 요약합니다.
4. 면접관이 질문할 가능성이 높은 내용이 있다면 포함합니다.
5. 내용이 적은 슬라이드는 억지로 내용을 만들어내지 않습니다.
6. 발표자료에 없는 사실을 추가하지 않습니다.
7. 한국어로 작성합니다.
8. 3~5문장 이내로 간결하게 작성합니다.

[출력]

요약 내용만 작성하세요.
"""

        try:

            response = (
                pt_interview_client
                .models
                .generate_content(
                    model=getattr(
                        settings,
                        "GEMINI_MODEL",
                        "gemini-3.5-flash-lite",
                    ),
                    contents=prompt,
                )
            )

            if (
                response is None
                or not response.text
            ):
                return self._fallback_slide_summary(
                    extracted_text
                )

            return response.text.strip()

        except Exception as e:

            print(
                f"Gemini PT slide summary error: {e}"
            )

            return self._fallback_slide_summary(
                extracted_text
            )

    def _fallback_summary(
        self,
        slides_text: str,
    ) -> str:

        """
        Gemini 전체 요약 실패 시 fallback.

        발표자료 텍스트 일부를 반환해서
        전체 분석 흐름이 중단되지 않게 한다.
        """

        text = slides_text.strip()

        if len(text) > 1000:
            text = text[:1000] + "..."

        return text

    def _fallback_slide_summary(
        self,
        extracted_text: str,
    ) -> str:

        """
        Gemini 슬라이드 요약 실패 시 fallback.
        """

        text = extracted_text.strip()

        if len(text) > 500:
            text = text[:500] + "..."

        return text