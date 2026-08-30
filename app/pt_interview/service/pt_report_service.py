import json
import re
from statistics import mean
from typing import Any

from app.core.config import settings
from app.core.gemini_client import create_gemini_client
from app.pt_interview.schema.pt_report_schema import (
    PtQuestionAnalysis,
    PtReportGenerateRequest,
    PtReportGenerateResponse,
)


class PtReportService:

    def __init__(self) -> None:
        self.client = create_gemini_client()

        self.model = getattr(
            settings,
            "GEMINI_MODEL",
            "gemini-3.5-flash-lite",
        )

    def generate_report(
        self,
        request: PtReportGenerateRequest,
    ) -> PtReportGenerateResponse:

        fallback_scores = (
            self._calculate_fallback_scores(
                request
            )
        )

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

            response = (
                self.client
                .models
                .generate_content(
                    model=self.model,
                    contents=prompt,
                )
            )

            response_text = getattr(
                response,
                "text",
                None,
            )

            print(
                "\n========== [PT REPORT] GEMINI RAW RESPONSE =========="
            )
            print(response_text)
            print(
                "======================================================\n"
            )

            if not response_text:
                raise ValueError(
                    "Gemini 응답 본문이 비어 있습니다."
                )

            parsed_result = (
                self._parse_json_response(
                    response_text
                )
            )

            return self._build_response(
                parsed_result=parsed_result,
                request=request,
                fallback_scores=fallback_scores,
            )

        except Exception as error:

            print(
                f"[PT REPORT] Gemini 분석 실패: {error}"
            )

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

        for item in request.qa_items:

            qa_items.append(
                {
                    "question_number":
                        item.question_number,
                    "question":
                        item.question,
                    "answer":
                        item.answer,
                }
            )

        input_data = {
            "presentation_material_summary":
                request.presentation_material_summary,

            "presentation_transcript":
                request.presentation_transcript,


            "qa_items":
                qa_items,

            "nonverbal_analysis": {
                "eye_scores":
                    request.eye_scores,
                "blink_scores":
                    request.blink_scores,
                "pose_scores":
                    request.pose_scores,
                "speech_scores":
                    request.speech_scores,
            },
            "calculated_reference_scores":
                fallback_scores,
        }

        return f"""
당신은 채용 PT 면접을 평가하는 전문 면접관입니다.

사용자가 업로드한 발표자료 자체의 내용과,
사용자가 실제 발표 영상에서 말한 내용,
면접관의 질문과 사용자의 실제 답변,
시선·눈 깜빡임·자세·음성 분석 결과를 종합하여
PT 면접 리포트를 작성하세요.

중요:
발표자료에 적힌 내용과 사용자가 실제로 말한 내용은
반드시 구분해서 평가해야 합니다.

발표자료에 좋은 내용이 있다고 해서
사용자가 실제 발표에서도 그것을 잘 설명했다고
판단하면 안 됩니다.

실제 발표 평가에서는 반드시
presentation_transcript를 우선적으로 사용하세요.


==================================================
1. content_score
==================================================

사용자가 실제 발표에서 전달한 내용을 평가합니다.

평가 요소:
- 발표 주제에 맞는 내용을 실제로 말했는가
- 핵심 주장을 명확하게 설명했는가
- 근거나 사례가 구체적인가
- 발표자료의 핵심 내용을 실제 발표에서도 설명했는가
- 단순히 자료에 내용이 존재한다는 이유로 높은 점수를 주지 않습니다.

0~100 정수입니다.


==================================================
2. structure_score
==================================================

사용자의 실제 발표 흐름을 평가합니다.

평가 요소:
- 도입 → 핵심 내용 → 결론의 흐름
- 문제 → 원인 → 해결 방안의 연결
- 문장과 문장 사이의 논리적 연결
- 핵심 메시지의 명확성
- 발표자료 순서를 그대로 읽었는지가 아니라
  실제 발표가 논리적으로 전달되었는지를 평가합니다.

0~100 정수입니다.


==================================================
3. delivery_score
==================================================

다음 비언어 및 음성 분석 결과를 종합합니다.

- eye_scores
- blink_scores
- pose_scores
- speech_scores

제공된 점수 평균을 주요 참고값으로 사용하세요.

0~100 정수입니다.


==================================================
4. qa_score
==================================================

모든 질의응답 결과를 종합한 점수입니다.

각 답변에 대해 다음을 평가합니다.

- 질문 의도를 이해했는가
- 질문에 직접 답했는가
- 논리적으로 설명했는가
- 구체적인 행동, 사례, 근거를 제시했는가
- 발표 내용과 일관성이 있는가
- 추상적인 표현만 반복하지 않았는가

답변이 비어 있거나 지나치게 짧으면
보수적으로 평가합니다.

0~100 정수입니다.


==================================================
5. overall_score
==================================================

다음 비율을 사용합니다.

content_score: 30%
structure_score: 20%
delivery_score: 25%
qa_score: 25%


==================================================
6. presentation_material_summary
==================================================

업로드된 PPT/PDF 자체의 핵심 내용을 요약합니다.

사용자가 실제 발표에서 말한 내용과
혼동하지 마세요.


==================================================
7. actual_presentation_summary
==================================================

presentation_transcript만을 중심으로
사용자가 실제로 말한 내용을 3~6문장으로 요약하세요.

발표자료에는 있지만 사용자가 말하지 않은 내용은
실제 발표 요약에 포함하지 마세요.

presentation_transcript가 비어 있다면
실제 발표 내용을 확인할 수 없다고 명시하세요.


==================================================
8. presentation_comparison
==================================================

발표자료와 실제 발표를 비교하여 분석하세요.

반드시 구체적으로 작성하세요.

예:
- 자료의 어떤 핵심 내용을 실제 발표에서도 설명했는지
- 자료에는 있지만 실제 발표에서 빠진 내용
- 실제 발표에서 추가적으로 설명한 내용
- 자료를 단순히 읽는 방식이었는지
- 핵심 근거 또는 결론이 실제 발표에서 충분했는지

단순히
"발표자료와 발표가 일치합니다"
라고 작성하지 마세요.


==================================================
9. question_analyses
==================================================

모든 질문마다 반드시 하나씩 생성하세요.

각 질문에 대해 다음 필드를 작성합니다.

question_number
- 입력받은 질문 번호

question
- 면접관이 실제로 한 질문
- 원문을 유지합니다.

answer
- 사용자의 실제 답변
- 입력된 답변 내용을 유지합니다.

understanding_score
- 질문 이해도
- 0~100

logic_score
- 답변 논리성
- 0~100

specificity_score
- 답변 구체성
- 0~100

evidence_score
- 근거, 사례, 발표자료 활용 정도
- 0~100

overall_score
- 해당 답변의 종합 점수
- 위 네 항목을 종합합니다.

strengths
- 이 답변에서 잘한 부분을
  실제 답변 내용을 근거로 구체적으로 설명합니다.

weaknesses
- 부족한 부분을 구체적으로 설명합니다.
- "구체성이 부족합니다"에서 끝내지 마세요.
- 무엇이 추상적이었는지 설명하세요.

improvement
- 어떻게 말하면 더 구체적이고 좋은 답변이 되는지
  행동 수준으로 설명하세요.

예:
나쁜 예:
"구체적인 사례를 추가하세요."

좋은 예:
"유아의 집중이 흐트러졌을 때 다시 집중시키겠다고만
말하기보다, 어떤 발문을 사용할지와 그래도 반응하지 않을 때
어떤 순서로 개입할지를 설명하면 좋습니다."

improved_answer
- 사용자의 기존 답변을 기반으로
  실제 면접에서 말할 수 있는 개선 답변 예시를 작성합니다.
- 3~7문장 정도로 작성합니다.
- 사용자가 말하지 않은 경험이나 사실을
  임의로 만들어내면 안 됩니다.
- 필요한 경우 조건형 표현을 사용하세요.
  예: "제가 실제 상황에 적용한다면..."


==================================================
10. strengths
==================================================

PT 면접 전체에서 잘한 점을 작성합니다.

반드시 실제 발표 또는 실제 답변의 특징을 근거로 작성하세요.


==================================================
11. weaknesses
==================================================

전체 면접에서 우선적으로 개선해야 할 부분을 작성합니다.

실제 발표와 질의응답을 구분해서 설명할 수 있습니다.


==================================================
12. improvement
==================================================

다음 PT 면접에서 바로 적용할 수 있는
구체적인 개선 방법을 제시하세요.

가능하면 다음 순서로 작성합니다.

1. 발표 내용
2. 발표 전달 방식
3. 질의응답


==================================================
입력 데이터
==================================================

{json.dumps(
    input_data,
    ensure_ascii=False,
    indent=2,
)}


==================================================
출력 규칙
==================================================

반드시 JSON 객체 하나만 반환하세요.

마크다운 코드 블록을 사용하지 마세요.
JSON 앞뒤에 설명을 추가하지 마세요.

아래 구조를 정확히 따르세요.

{{
  "content_score": 0,
  "structure_score": 0,
  "delivery_score": 0,
  "qa_score": 0,
  "overall_score": 0,

  "presentation_material_summary": "",
  "actual_presentation_summary": "",
  "presentation_comparison": "",

  "question_analyses": [
    {{
      "question_number": 1,
      "question": "",
      "answer": "",
      "understanding_score": 0,
      "logic_score": 0,
      "specificity_score": 0,
      "evidence_score": 0,
      "overall_score": 0,
      "strengths": "",
      "weaknesses": "",
      "improvement": "",
      "improved_answer": ""
    }}
  ],

  "strengths": "",
  "weaknesses": "",
  "improvement": ""
}}
""".strip()

    def _calculate_fallback_scores(
        self,
        request: PtReportGenerateRequest,
    ) -> dict[str, int]:

        content_score = (
            self._calculate_content_score(
                request
            )
        )

        structure_score = (
            self._calculate_structure_score(
                request
            )
        )

        delivery_score = (
            self._calculate_delivery_score(
                request
            )
        )

        qa_score = (
            self._calculate_qa_score(
                request
            )
        )

        overall_score = round(
            content_score * 0.30
            + structure_score * 0.20
            + delivery_score * 0.25
            + qa_score * 0.25
        )

        return {
            "content_score":
                self._normalize_score(
                    content_score
                ),
            "structure_score":
                self._normalize_score(
                    structure_score
                ),
            "delivery_score":
                self._normalize_score(
                    delivery_score
                ),
            "qa_score":
                self._normalize_score(
                    qa_score
                ),
            "overall_score":
                self._normalize_score(
                    overall_score
                ),
        }

    def _calculate_content_score(
        self,
        request: PtReportGenerateRequest,
    ) -> int:

        transcript = (
            request
            .presentation_transcript
            .strip()
        )

        if transcript:

            transcript_length = len(
                transcript
            )

            if transcript_length >= 1000:
                return 85

            if transcript_length >= 600:
                return 80

            if transcript_length >= 300:
                return 72

            if transcript_length >= 100:
                return 62

            return 50

        material = (
            request
            .presentation_material_summary
            .strip()
        )

        if material:
            return 50

        return 40

    def _calculate_structure_score(
        self,
        request: PtReportGenerateRequest,
    ) -> int:

        transcript = (
            request
            .presentation_transcript
            .strip()
        )

        if not transcript:
            return 45

        structure_keywords = [
            "먼저",
            "첫째",
            "둘째",
            "셋째",
            "다음으로",
            "따라서",
            "그러므로",
            "결론적으로",
            "마지막으로",
            "문제",
            "원인",
            "해결",
            "개선",
        ]

        matched_count = sum(
            1
            for keyword in structure_keywords
            if keyword in transcript
        )

        score = (
            55
            + min(
                matched_count * 4,
                30,
            )
        )

        if len(transcript) >= 500:
            score += 5

        return self._normalize_score(
            score
        )

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
            self._normalize_float_score(
                score
            )
            for score in all_scores
        ]

        if not valid_scores:
            return 50

        return round(
            mean(valid_scores)
        )

    def _calculate_qa_score(
        self,
        request: PtReportGenerateRequest,
    ) -> int:

        if not request.qa_items:
            return 40

        answer_scores = []

        for item in request.qa_items:

            answer = (
                item.answer
                .strip()
            )

            answer_length = len(
                answer
            )

            if answer_length >= 300:
                score = 85

            elif answer_length >= 150:
                score = 78

            elif answer_length >= 70:
                score = 68

            elif answer_length >= 30:
                score = 58

            elif answer_length > 0:
                score = 48

            else:
                score = 25

            answer_scores.append(
                score
            )

        if not answer_scores:
            return 40

        answered_count = sum(
            1
            for item in request.qa_items
            if item.answer.strip()
        )

        answered_ratio = (
            answered_count
            / len(request.qa_items)
        )

        average_answer_score = (
            mean(answer_scores)
        )

        result = round(
            average_answer_score * 0.80
            + answered_ratio * 100 * 0.20
        )

        return self._normalize_score(
            result
        )

    def _parse_json_response(
        self,
        response_text: str,
    ) -> dict[str, Any]:

        cleaned_text = (
            response_text
            .strip()
        )

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

            return json.loads(
                cleaned_text
            )

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

            return json.loads(
                json_match.group()
            )

    def _build_response(
        self,
        parsed_result: dict[str, Any],
        request: PtReportGenerateRequest,
        fallback_scores: dict[str, int],
    ) -> PtReportGenerateResponse:

        content_score = self._get_score(
            parsed_result,
            "content_score",
            fallback_scores[
                "content_score"
            ],
        )

        structure_score = self._get_score(
            parsed_result,
            "structure_score",
            fallback_scores[
                "structure_score"
            ],
        )

        delivery_score = self._get_score(
            parsed_result,
            "delivery_score",
            fallback_scores[
                "delivery_score"
            ],
        )

        qa_score = self._get_score(
            parsed_result,
            "qa_score",
            fallback_scores[
                "qa_score"
            ],
        )

        calculated_overall_score = round(
            content_score * 0.30
            + structure_score * 0.20
            + delivery_score * 0.25
            + qa_score * 0.25
        )

        question_analyses = (
            self._build_question_analyses(
                parsed_result=parsed_result,
                request=request,
            )
        )

        material_summary = str(
            parsed_result.get(
                "presentation_material_summary"
            )
            or request.presentation_material_summary
            or "발표자료 요약 정보가 제공되지 않았습니다."
        )

        actual_summary = str(
            parsed_result.get(
                "actual_presentation_summary"
            )
            or self._fallback_actual_summary(
                request
            )
        )

        comparison = str(
            parsed_result.get(
                "presentation_comparison"
            )
            or self._fallback_comparison(
                request
            )
        )

        return PtReportGenerateResponse(
            content_score=
                content_score,

            structure_score=
                structure_score,

            delivery_score=
                delivery_score,

            qa_score=
                qa_score,

            overall_score=
                self._normalize_score(
                    calculated_overall_score
                ),

            presentation_material_summary=
                material_summary,

            actual_presentation_summary=
                actual_summary,

            presentation_comparison=
                comparison,

            question_analyses=
                question_analyses,

            strengths=str(
                parsed_result.get(
                    "strengths"
                )
                or (
                    "실제 발표와 질의응답을 "
                    "바탕으로 핵심 내용을 "
                    "전달했습니다."
                )
            ),

            weaknesses=str(
                parsed_result.get(
                    "weaknesses"
                )
                or (
                    "발표 내용의 구조화와 "
                    "질의응답의 구체성을 "
                    "추가로 보완할 필요가 있습니다."
                )
            ),

            improvement=str(
                parsed_result.get(
                    "improvement"
                )
                or (
                    "발표에서는 핵심 주장과 근거를 "
                    "연결하고, 질의응답에서는 "
                    "구체적인 행동이나 사례까지 "
                    "설명하는 방식으로 답변하세요."
                )
            ),
        )

    def _build_question_analyses(
        self,
        parsed_result: dict[str, Any],
        request: PtReportGenerateRequest,
    ) -> list[PtQuestionAnalysis]:

        raw_analyses = (
            parsed_result.get(
                "question_analyses"
            )
            or []
        )

        result: list[
            PtQuestionAnalysis
        ] = []

        for index, item in enumerate(
            request.qa_items
        ):

            parsed_item = {}

            if (
                index
                < len(raw_analyses)
                and isinstance(
                    raw_analyses[index],
                    dict,
                )
            ):

                parsed_item = (
                    raw_analyses[index]
                )

            fallback_score = (
                self._calculate_single_answer_score(
                    item.answer
                )
            )

            result.append(
                PtQuestionAnalysis(
                    question_number=
                        item.question_number,

                    question=
                        item.question,

                    answer=
                        item.answer,

                    understanding_score=
                        self._get_score(
                            parsed_item,
                            "understanding_score",
                            fallback_score,
                        ),

                    logic_score=
                        self._get_score(
                            parsed_item,
                            "logic_score",
                            fallback_score,
                        ),

                    specificity_score=
                        self._get_score(
                            parsed_item,
                            "specificity_score",
                            max(
                                0,
                                fallback_score - 5,
                            ),
                        ),

                    evidence_score=
                        self._get_score(
                            parsed_item,
                            "evidence_score",
                            max(
                                0,
                                fallback_score - 5,
                            ),
                        ),

                    overall_score=
                        self._get_score(
                            parsed_item,
                            "overall_score",
                            fallback_score,
                        ),

                    strengths=str(
                        parsed_item.get(
                            "strengths"
                        )
                        or self._fallback_qa_strength(
                            item.answer
                        )
                    ),

                    weaknesses=str(
                        parsed_item.get(
                            "weaknesses"
                        )
                        or self._fallback_qa_weakness(
                            item.answer
                        )
                    ),

                    improvement=str(
                        parsed_item.get(
                            "improvement"
                        )
                        or (
                            "질문의 핵심에 먼저 답한 뒤 "
                            "구체적인 이유, 행동 또는 "
                            "사례를 덧붙여 설명하세요."
                        )
                    ),

                    improved_answer=str(
                        parsed_item.get(
                            "improved_answer"
                        )
                        or self._fallback_improved_answer(
                            item.question,
                            item.answer,
                        )
                    ),
                )
            )

        return result

    def _create_fallback_response(
        self,
        request: PtReportGenerateRequest,
        scores: dict[str, int],
    ) -> PtReportGenerateResponse:

        question_analyses = []

        for item in request.qa_items:

            score = (
                self._calculate_single_answer_score(
                    item.answer
                )
            )

            question_analyses.append(
                PtQuestionAnalysis(
                    question_number=
                        item.question_number,

                    question=
                        item.question,

                    answer=
                        item.answer,

                    understanding_score=
                        score,

                    logic_score=
                        score,

                    specificity_score=
                        max(
                            0,
                            score - 5,
                        ),

                    evidence_score=
                        max(
                            0,
                            score - 5,
                        ),

                    overall_score=
                        score,

                    strengths=
                        self._fallback_qa_strength(
                            item.answer
                        ),

                    weaknesses=
                        self._fallback_qa_weakness(
                            item.answer
                        ),

                    improvement=(
                        "질문의 핵심에 먼저 답하고, "
                        "그 이유와 실제로 취할 행동을 "
                        "순서대로 설명하세요."
                    ),

                    improved_answer=
                        self._fallback_improved_answer(
                            item.question,
                            item.answer,
                        ),
                )
            )

        return PtReportGenerateResponse(
            content_score=
                scores["content_score"],

            structure_score=
                scores["structure_score"],

            delivery_score=
                scores["delivery_score"],

            qa_score=
                scores["qa_score"],

            overall_score=
                scores["overall_score"],

            presentation_material_summary=(
                request
                .presentation_material_summary
                .strip()
                or (
                    "발표자료 요약 정보가 "
                    "제공되지 않았습니다."
                )
            ),

            actual_presentation_summary=
                self._fallback_actual_summary(
                    request
                ),

            presentation_comparison=
                self._fallback_comparison(
                    request
                ),

            question_analyses=
                question_analyses,

            strengths=(
                "실제 발표와 질의응답을 통해 "
                "주제의 핵심 내용을 전달하려고 했습니다."
            ),

            weaknesses=(
                "발표의 구조와 질의응답의 "
                "구체성을 추가로 보완할 필요가 있습니다."
            ),

            improvement=(
                "발표에서는 핵심 주장과 근거를 "
                "도입-본론-결론 순서로 정리하고, "
                "질문에는 결론을 먼저 말한 뒤 "
                "이유와 구체적인 사례 또는 행동을 "
                "덧붙이세요."
            ),
        )

    def _fallback_actual_summary(
        self,
        request: PtReportGenerateRequest,
    ) -> str:

        transcript = (
            request
            .presentation_transcript
            .strip()
        )

        if not transcript:

            return (
                "실제 발표 음성의 STT 결과가 없어 "
                "사용자가 실제로 말한 내용을 "
                "확인할 수 없습니다."
            )

        if len(transcript) > 1000:

            return (
                transcript[:1000]
                + "..."
            )

        return transcript

    def _fallback_comparison(
        self,
        request: PtReportGenerateRequest,
    ) -> str:

        material = (
            request
            .presentation_material_summary
            .strip()
        )

        transcript = (
            request
            .presentation_transcript
            .strip()
        )

        if not material and not transcript:

            return (
                "발표자료와 실제 발표 내용이 없어 "
                "비교 분석을 수행할 수 없습니다."
            )

        if not transcript:

            return (
                "발표자료 분석 결과는 존재하지만 "
                "실제 발표 STT가 없어 자료와 실제 발표를 "
                "비교할 수 없습니다."
            )

        if not material:

            return (
                "실제 발표 STT는 존재하지만 "
                "발표자료 요약이 없어 자료와 실제 발표를 "
                "비교할 수 없습니다."
            )

        return (
            "발표자료와 실제 발표 내용이 모두 존재합니다. "
            "Gemini 분석을 사용할 수 없는 상태이므로 "
            "세부적인 누락·추가 내용 비교는 제공하지 않습니다."
        )

    def _fallback_qa_strength(
        self,
        answer: str,
    ) -> str:

        if not answer.strip():

            return (
                "실제 답변 내용이 없어 "
                "강점을 평가하기 어렵습니다."
            )

        return (
            "질문에 대해 실제 답변을 제시했다는 점은 "
            "확인할 수 있습니다."
        )

    def _fallback_qa_weakness(
        self,
        answer: str,
    ) -> str:

        text = answer.strip()

        if not text:

            return (
                "답변이 없어 질문에 대한 "
                "대응 능력을 평가할 수 없습니다."
            )

        if len(text) < 70:

            return (
                "답변이 짧아 이유, 근거 또는 "
                "구체적인 행동 설명이 충분하지 않습니다."
            )

        return (
            "답변의 핵심 주장과 이를 뒷받침하는 "
            "구체적인 근거를 보다 명확하게 "
            "연결할 필요가 있습니다."
        )

    def _fallback_improved_answer(
        self,
        question: str,
        answer: str,
    ) -> str:

        if not answer.strip():

            return (
                "질문의 핵심에 대한 결론을 먼저 말하고, "
                "그 판단의 이유와 실제 상황에서 취할 행동을 "
                "구체적인 순서로 설명하는 방식으로 "
                "답변하는 것이 좋습니다."
            )

        return (
            "현재 답변의 핵심 내용을 먼저 한 문장으로 "
            "정리한 뒤, 그렇게 판단한 이유를 설명하고 "
            "실제 상황에서 취할 구체적인 행동이나 "
            "예시를 덧붙이는 방식으로 답변하면 좋습니다."
        )

    def _calculate_single_answer_score(
        self,
        answer: str,
    ) -> int:

        answer_length = len(
            answer.strip()
        )

        if answer_length >= 300:
            return 85

        if answer_length >= 150:
            return 78

        if answer_length >= 70:
            return 68

        if answer_length >= 30:
            return 58

        if answer_length > 0:
            return 48

        return 25

    def _get_score(
        self,
        result: dict[str, Any],
        key: str,
        default: int,
    ) -> int:

        try:

            return self._normalize_score(
                float(
                    result.get(
                        key,
                        default,
                    )
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            return default

    @staticmethod
    def _normalize_score(
        score: float,
    ) -> int:

        return max(
            0,
            min(
                100,
                round(score),
            ),
        )

    @staticmethod
    def _normalize_float_score(
        score: float,
    ) -> float:

        try:

            return max(
                0.0,
                min(
                    100.0,
                    float(score),
                ),
            )

        except (
            TypeError,
            ValueError,
        ):

            return 0.0