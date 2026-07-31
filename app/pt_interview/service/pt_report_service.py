import json
import re
from statistics import mean
from typing import Any

from app.core.gemini_client import create_gemini_client
from app.pt_interview.schema.pt_report_schema import (
    PtReportGenerateRequest,
    PtReportGenerateResponse,
)


class PtReportService:

    def __init__(self) -> None:
        self.client = create_gemini_client()
        self.model = "gemini-2.5-flash-lite"

    def generate_report(
        self,
        request: PtReportGenerateRequest,
    ) -> PtReportGenerateResponse:
        fallback_scores = self._calculate_fallback_scores(request)

        if self.client is None:
            return self._create_fallback_response(
                request=request,
                scores=fallback_scores,
            )

        prompt = self._build_prompt(
            request=request,
            fallback_scores=fallback_scores,
        )

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
            )

            response_text = getattr(response, "text", None)

            if not response_text:
                raise ValueError("Gemini 응답 본문이 비어 있습니다.")

            parsed_result = self._parse_json_response(response_text)

            return self._build_response(
                parsed_result=parsed_result,
                request=request,
                fallback_scores=fallback_scores,
            )

        except Exception as error:
            print(f"[PT REPORT] Gemini 분석 실패: {error}")

            return self._create_fallback_response(
                request=request,
                scores=fallback_scores,
            )

    def _build_prompt(
        self,
        request: PtReportGenerateRequest,
        fallback_scores: dict[str, int],
    ) -> str:
        qa_items = []

        max_length = max(
            len(request.questions),
            len(request.answers),
        )

        for index in range(max_length):
            question = (
                request.questions[index]
                if index < len(request.questions)
                else ""
            )
            answer = (
                request.answers[index]
                if index < len(request.answers)
                else ""
            )

            qa_items.append(
                {
                    "question_number": index + 1,
                    "question": question,
                    "answer": answer,
                }
            )

        input_data = {
            "presentation_summary": request.presentation_summary,
            "qa_items": qa_items,
            "nonverbal_analysis": {
                "eye_scores": request.eye_scores,
                "blink_scores": request.blink_scores,
                "pose_scores": request.pose_scores,
                "speech_scores": request.speech_scores,
            },
            "calculated_reference_scores": fallback_scores,
        }

        return f"""
당신은 채용 PT 면접을 평가하는 전문 면접관입니다.

아래 발표 자료 분석 결과, 면접 질문과 답변, 비언어적 분석 점수를
종합하여 PT 면접 리포트를 작성하세요.

평가 기준은 다음과 같습니다.

1. content_score
- 발표 내용이 주제에 적합한지 평가합니다.
- 핵심 내용의 정확성, 구체성, 직무 관련성을 평가합니다.
- 0점에서 100점 사이의 정수입니다.

2. structure_score
- 발표 내용이 논리적으로 구성되어 있는지 평가합니다.
- 도입, 본론, 결론의 연결성과 핵심 메시지의 명확성을 평가합니다.
- 0점에서 100점 사이의 정수입니다.

3. delivery_score
- 시선, 눈 깜빡임, 자세, 발화 분석 결과를 종합합니다.
- 제공된 점수들의 평균을 참고하되 발표 전달력 관점에서 평가합니다.
- 0점에서 100점 사이의 정수입니다.

4. qa_score
- 질문의 의도를 이해했는지 평가합니다.
- 답변의 정확성, 논리성, 구체성, 발표 내용과의 일관성을 평가합니다.
- 0점에서 100점 사이의 정수입니다.
- 질문이나 답변이 없다면 보수적으로 평가합니다.

5. overall_score
- content_score 30%
- structure_score 20%
- delivery_score 25%
- qa_score 25%
위 비율을 기준으로 계산한 종합 점수입니다.

6. presentation_summary
- 발표 전체 내용을 3~5문장으로 요약합니다.

7. strengths
- 잘한 점을 구체적으로 작성합니다.
- 단순히 점수가 높다고 하지 말고 발표 내용이나 답변 특징을 설명합니다.

8. weaknesses
- 부족한 점을 구체적으로 작성합니다.
- 공격적 표현은 피하고 개선할 수 있는 문제를 중심으로 작성합니다.

9. improvement
- 다음 PT 면접에서 적용할 수 있는 개선 방법을 구체적으로 작성합니다.

입력 데이터:
{json.dumps(input_data, ensure_ascii=False, indent=2)}

반드시 아래 JSON 형식만 반환하세요.
마크다운 코드 블록이나 추가 설명을 포함하지 마세요.

{{
  "content_score": 0,
  "structure_score": 0,
  "delivery_score": 0,
  "qa_score": 0,
  "overall_score": 0,
  "presentation_summary": "",
  "strengths": "",
  "weaknesses": "",
  "improvement": ""
}}
""".strip()

    def _calculate_fallback_scores(
        self,
        request: PtReportGenerateRequest,
    ) -> dict[str, int]:
        content_score = self._calculate_content_score(request)
        structure_score = self._calculate_structure_score(request)
        delivery_score = self._calculate_delivery_score(request)
        qa_score = self._calculate_qa_score(request)

        overall_score = round(
            content_score * 0.30
            + structure_score * 0.20
            + delivery_score * 0.25
            + qa_score * 0.25
        )

        return {
            "content_score": self._normalize_score(content_score),
            "structure_score": self._normalize_score(structure_score),
            "delivery_score": self._normalize_score(delivery_score),
            "qa_score": self._normalize_score(qa_score),
            "overall_score": self._normalize_score(overall_score),
        }

    def _calculate_content_score(
        self,
        request: PtReportGenerateRequest,
    ) -> int:
        summary_length = len(request.presentation_summary.strip())

        if summary_length >= 500:
            return 85
        if summary_length >= 250:
            return 78
        if summary_length >= 100:
            return 70
        if summary_length > 0:
            return 60

        return 40

    def _calculate_structure_score(
        self,
        request: PtReportGenerateRequest,
    ) -> int:
        summary = request.presentation_summary.strip()

        if not summary:
            return 40

        structure_keywords = [
            "첫째",
            "둘째",
            "마지막",
            "결론",
            "요약",
            "배경",
            "문제",
            "해결",
            "기대 효과",
        ]

        matched_count = sum(
            1 for keyword in structure_keywords
            if keyword in summary
        )

        score = 60 + min(matched_count * 4, 25)

        if len(summary) >= 300:
            score += 5

        return self._normalize_score(score)

    def _calculate_delivery_score(
        self,
        request: PtReportGenerateRequest,
    ) -> int:
        all_scores = [
            *request.eye_scores,
            *request.blink_scores,
            *request.pose_scores,
            *request.speech_scores,
        ]

        valid_scores = [
            self._normalize_float_score(score)
            for score in all_scores
        ]

        if not valid_scores:
            return 50

        return round(mean(valid_scores))

    def _calculate_qa_score(
        self,
        request: PtReportGenerateRequest,
    ) -> int:
        if not request.questions or not request.answers:
            return 40

        answer_scores = []

        for answer in request.answers:
            answer_length = len(answer.strip())

            if answer_length >= 300:
                answer_scores.append(85)
            elif answer_length >= 150:
                answer_scores.append(78)
            elif answer_length >= 70:
                answer_scores.append(68)
            elif answer_length > 0:
                answer_scores.append(55)
            else:
                answer_scores.append(30)

        answered_ratio = min(
            len(request.answers) / max(len(request.questions), 1),
            1.0,
        )

        average_answer_score = mean(answer_scores)

        return round(
            average_answer_score * 0.8
            + answered_ratio * 100 * 0.2
        )

    def _parse_json_response(
        self,
        response_text: str,
    ) -> dict[str, Any]:
        cleaned_text = response_text.strip()

        cleaned_text = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned_text,
            flags=re.IGNORECASE,
        )
        cleaned_text = re.sub(
            r"\s*```$",
            "",
            cleaned_text,
        )

        try:
            return json.loads(cleaned_text)

        except json.JSONDecodeError:
            json_match = re.search(
                r"\{.*\}",
                cleaned_text,
                flags=re.DOTALL,
            )

            if not json_match:
                raise ValueError(
                    "Gemini 응답에서 JSON 객체를 찾을 수 없습니다."
                )

            return json.loads(json_match.group())

    def _build_response(
        self,
        parsed_result: dict[str, Any],
        request: PtReportGenerateRequest,
        fallback_scores: dict[str, int],
    ) -> PtReportGenerateResponse:
        content_score = self._get_score(
            parsed_result,
            "content_score",
            fallback_scores["content_score"],
        )
        structure_score = self._get_score(
            parsed_result,
            "structure_score",
            fallback_scores["structure_score"],
        )
        delivery_score = self._get_score(
            parsed_result,
            "delivery_score",
            fallback_scores["delivery_score"],
        )
        qa_score = self._get_score(
            parsed_result,
            "qa_score",
            fallback_scores["qa_score"],
        )

        calculated_overall_score = round(
            content_score * 0.30
            + structure_score * 0.20
            + delivery_score * 0.25
            + qa_score * 0.25
        )

        presentation_summary = str(
            parsed_result.get("presentation_summary")
            or request.presentation_summary
            or "발표 요약 정보가 제공되지 않았습니다."
        )

        return PtReportGenerateResponse(
            content_score=content_score,
            structure_score=structure_score,
            delivery_score=delivery_score,
            qa_score=qa_score,
            overall_score=self._normalize_score(
                calculated_overall_score
            ),
            presentation_summary=presentation_summary,
            strengths=str(
                parsed_result.get("strengths")
                or "발표 자료와 답변을 기반으로 강점을 추가 분석해야 합니다."
            ),
            weaknesses=str(
                parsed_result.get("weaknesses")
                or "발표 자료와 답변을 기반으로 보완점을 추가 분석해야 합니다."
            ),
            improvement=str(
                parsed_result.get("improvement")
                or "발표 구조를 명확히 하고 예상 질문에 대한 구체적인 답변을 준비하세요."
            ),
        )

    def _create_fallback_response(
        self,
        request: PtReportGenerateRequest,
        scores: dict[str, int],
    ) -> PtReportGenerateResponse:
        summary = (
            request.presentation_summary.strip()
            or "발표 요약 정보가 제공되지 않았습니다."
        )

        return PtReportGenerateResponse(
            content_score=scores["content_score"],
            structure_score=scores["structure_score"],
            delivery_score=scores["delivery_score"],
            qa_score=scores["qa_score"],
            overall_score=scores["overall_score"],
            presentation_summary=summary,
            strengths=(
                "발표 자료와 질의응답 내용을 바탕으로 핵심 내용을 전달했습니다."
            ),
            weaknesses=(
                "발표 내용의 구조화와 답변의 구체성을 추가로 보완할 필요가 있습니다."
            ),
            improvement=(
                "도입, 핵심 근거, 결론 순서로 발표 내용을 정리하고 "
                "예상 질문에는 근거와 사례를 포함해 답변하세요."
            ),
        )

    def _get_score(
        self,
        result: dict[str, Any],
        key: str,
        default: int,
    ) -> int:
        try:
            return self._normalize_score(
                float(result.get(key, default))
            )
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _normalize_score(score: float) -> int:
        return max(0, min(100, round(score)))

    @staticmethod
    def _normalize_float_score(score: float) -> float:
        try:
            return max(0.0, min(100.0, float(score)))
        except (TypeError, ValueError):
            return 0.0