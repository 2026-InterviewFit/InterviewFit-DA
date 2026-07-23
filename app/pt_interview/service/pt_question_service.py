import json

from app.core.config import settings
from app.core.gemini_client import pt_interview_client

from app.pt_interview.prompts.pt_question_prompts import (
    build_pt_question_prompt,
)
from app.pt_interview.schema.pt_question_schema import (
    PtQuestionGenerateResponse,
    PtQuestionResult,
)


class PtQuestionService:

    def generate_questions(
        self,
        document_summary: str,
        slides_text: str,
        question_count: int = 5,
    ) -> PtQuestionGenerateResponse:

        if not document_summary.strip():
            raise ValueError(
                "발표자료 전체 요약이 비어 있습니다."
            )

        if pt_interview_client is None:

            return self._fallback_questions(
                question_count
            )

        prompt = build_pt_question_prompt(
            document_summary=document_summary,
            slides_text=slides_text,
            question_count=question_count,
        )

        try:

            response = (
                pt_interview_client
                .models
                .generate_content(
                    model=getattr(
                        settings,
                        "GEMINI_MODEL",
                        "gemini-2.5-flash-lite",
                    ),
                    contents=prompt,
                )
            )

            if (
                response is None
                or not response.text
            ):
                return self._fallback_questions(
                    question_count
                )

            data = self._parse_json(
                response.text
            )

            questions = []

            for index, item in enumerate(
                data.get("questions", []),
                start=1,
            ):

                question = str(
                    item.get(
                        "question",
                        "",
                    )
                ).strip()

                if not question:
                    continue

                questions.append(
                    PtQuestionResult(
                        question_number=item.get(
                            "question_number",
                            index,
                        ),
                        question=question,
                        reason=item.get(
                            "reason"
                        ),
                    )
                )

            if not questions:

                return self._fallback_questions(
                    question_count
                )

            return PtQuestionGenerateResponse(
                questions=questions[
                    :question_count
                ]
            )

        except Exception as e:

            print(
                f"Gemini PT question error: {e}"
            )

            return self._fallback_questions(
                question_count
            )

    def _parse_json(
        self,
        response_text: str,
    ) -> dict:

        text = response_text.strip()

        if text.startswith("```json"):

            text = text[
                len("```json"):
            ]

        elif text.startswith("```"):

            text = text[
                len("```"):
            ]

        if text.endswith("```"):

            text = text[:-3]

        return json.loads(
            text.strip()
        )

    def _fallback_questions(
        self,
        question_count: int,
    ) -> PtQuestionGenerateResponse:

        fallback = [
            "발표에서 제안한 핵심 방안을 선택한 이유는 무엇인가요?",
            "제안한 방안을 실제로 적용할 때 가장 큰 한계는 무엇이라고 생각하나요?",
            "발표에서 제시한 근거가 충분하다고 판단한 이유는 무엇인가요?",
            "다른 대안과 비교했을 때 이 방안의 가장 큰 장점은 무엇인가요?",
            "실제 실행 과정에서 예상과 다른 결과가 발생한다면 어떻게 대응하시겠습니까?",
        ]

        questions = []

        for index, question in enumerate(
            fallback[:question_count],
            start=1,
        ):

            questions.append(
                PtQuestionResult(
                    question_number=index,
                    question=question,
                    reason="기본 PT 면접 질문",
                )
            )

        return PtQuestionGenerateResponse(
            questions=questions
        )