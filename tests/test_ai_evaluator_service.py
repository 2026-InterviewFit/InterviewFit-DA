import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.common.exceptions.custom_exception import CustomException
from app.common.exceptions.error_code import ErrorCode
from app.speech_coaching.schemas.speech_schema import ExternalAiAnalysis
from app.speech_coaching.services import ai_evaluator_service


VALID_DATA = {
    "star_score": {"S": 8, "T": 7, "A": 9, "R": 8},
    "feedback": "STAR 구조가 명확합니다.",
    "keywords": ["협업", "문제해결"],
    "model_answer": "상황과 성과를 연결한 모범 답변입니다.",
}

VALID_EXTERNAL_DATA = {
    "star": {
        "situation": "문제 상황을 설명했습니다.",
        "task": "목표를 제시했습니다.",
        "action": "행동을 구체적으로 설명했습니다.",
        "result": "성과를 제시했습니다.",
    },
    "strengths": ["행동이 구체적입니다."],
    "improvements": ["성과를 수치화하면 좋습니다."],
}


def _json_text(feedback: str = "STAR 구조가 명확합니다.") -> str:
    return (
        '{"star_score":{"S":8,"T":7,"A":9,"R":8},'
        f'"feedback":"{feedback}",'
        '"keywords":["협업","문제해결"],'
        '"model_answer":"상황과 성과를 연결한 모범 답변입니다."}'
    )


class AiEvaluatorServiceTest(unittest.TestCase):
    def _evaluate_text(self, text: str, script: str = "답변"):
        response = SimpleNamespace(parsed=None, text=text)
        with patch.object(
            ai_evaluator_service.client.models,
            "generate_content",
            return_value=response,
        ) as generate_content:
            result = ai_evaluator_service.evaluate_interview_content(
                script,
                "백엔드",
                "입사 후 원하지 않는 부서에 배치된다면 어떻게 하겠습니까?",
            )

        config = generate_content.call_args.kwargs["config"]
        self.assertEqual(config.response_mime_type, "application/json")
        self.assertIs(
            config.response_schema,
            ai_evaluator_service.GeminiAiEvaluation,
        )
        return result

    def test_question_and_field_roles_are_included_for_answer_variants(self):
        question = "입사 후 원하지 않는 부서에 배치된다면 어떻게 하겠습니까?"
        cases = [
            ("회사 배치 이유를 이해하고 새로운 업무에 적응하겠습니다.", False),
            ("잘 모르겠습니다.", False),
            ("", True),
        ]
        response = SimpleNamespace(
            parsed=ai_evaluator_service.GeminiAiEvaluation.model_validate(VALID_DATA),
            text=_json_text(),
        )

        for script, is_empty in cases:
            with self.subTest(script=script):
                with patch.object(
                    ai_evaluator_service.client.models,
                    "generate_content",
                    return_value=response,
                ) as generate_content:
                    result = ai_evaluator_service.evaluate_interview_content(
                        script,
                        "백엔드",
                        question,
                    )

                prompt = generate_content.call_args.kwargs["contents"]
                self.assertIn(f"면접 질문 원문: {question}", prompt)
                self.assertIn("feedback: 지원자가 실제로 말한 답변", prompt)
                self.assertIn("model_answer: 아래 '면접 질문 원문'에 직접 답하는", prompt)
                self.assertIn("자연스러운 1인칭 답변", prompt)
                self.assertIn("평가는 feedback에만 작성", prompt)
                if is_empty:
                    self.assertIn("지원자가 실제로 말한 답변: (빈 답변)", prompt)
                else:
                    self.assertIn(f"지원자가 실제로 말한 답변: {script}", prompt)
                self.assertNotIn("[외부 면접 답변 평가 AI의 참고 분석]", prompt)
                self.assertNotEqual(result["feedback"], result["model_answer"])

    def test_external_analysis_is_added_without_replacing_question_or_answer(self):
        question = "갈등을 해결한 경험을 말씀해 주세요."
        answer = "의견 차이를 정리하고 합의안을 제시했습니다."
        external_analysis = ExternalAiAnalysis.model_validate(VALID_EXTERNAL_DATA)
        response = SimpleNamespace(
            parsed=ai_evaluator_service.GeminiAiEvaluation.model_validate(VALID_DATA),
            text=_json_text(),
        )

        with patch.object(
            ai_evaluator_service.client.models,
            "generate_content",
            return_value=response,
        ) as generate_content:
            ai_evaluator_service.evaluate_interview_content(
                answer,
                "백엔드",
                question,
                external_analysis,
            )

        prompt = generate_content.call_args.kwargs["contents"]
        self.assertIn(f"면접 질문 원문: {question}", prompt)
        self.assertIn(f"지원자가 실제로 말한 답변: {answer}", prompt)
        self.assertIn("[외부 면접 답변 평가 AI의 참고 분석]", prompt)
        self.assertIn('\"situation\": \"문제 상황을 설명했습니다.\"', prompt)
        self.assertIn("원본 면접 질문과 지원자가 실제로 말한 답변을 최우선", prompt)
        self.assertIn("외부 AI 결과를 그대로 복사하지 말고", prompt)
        self.assertIn("충돌하면 원본 질문과 실제 답변을 우선", prompt)

    def test_gemini_schema_has_no_additional_properties(self):
        schema = ai_evaluator_service.GeminiAiEvaluation.model_json_schema()

        def contains_additional_properties(value):
            if isinstance(value, dict):
                return "additionalProperties" in value or any(
                    contains_additional_properties(child)
                    for child in value.values()
                )
            if isinstance(value, list):
                return any(contains_additional_properties(child) for child in value)
            return False

        self.assertFalse(contains_additional_properties(schema))

    def test_structured_output_converts_to_public_response_shape(self):
        parsed = ai_evaluator_service.GeminiAiEvaluation.model_validate(VALID_DATA)
        response = SimpleNamespace(parsed=parsed, text=_json_text())
        with patch.object(
            ai_evaluator_service.client.models,
            "generate_content",
            return_value=response,
        ):
            result = ai_evaluator_service.evaluate_interview_content("답변", "백엔드")

        self.assertEqual(result, VALID_DATA)
        self.assertEqual(result["star_score"], {"S": 8, "T": 7, "A": 9, "R": 8})
        self.assertEqual(set(result), set(VALID_DATA))

    def test_gemini_api_failure_raises_ai_evaluation_failed(self):
        with patch.object(
            ai_evaluator_service.client.models,
            "generate_content",
            side_effect=ValueError("API failure"),
        ):
            with self.assertRaises(CustomException) as raised:
                ai_evaluator_service.evaluate_interview_content("답변", "백엔드")

        self.assertEqual(raised.exception.code, ErrorCode.AI_EVALUATION_FAILED)

    def test_normal_json_response(self):
        self.assertEqual(self._evaluate_text(_json_text()), VALID_DATA)

    def test_markdown_json_code_fence(self):
        result = self._evaluate_text(f"```json\n{_json_text()}\n```")
        self.assertEqual(result, VALID_DATA)

    def test_explanation_after_json(self):
        result = self._evaluate_text(f"{_json_text()}\n위와 같이 평가했습니다.")
        self.assertEqual(result, VALID_DATA)

    def test_literal_newline_inside_json_string(self):
        result = self._evaluate_text(_json_text("첫 번째 문장\n두 번째 문장"))
        self.assertEqual(result["feedback"], "첫 번째 문장\n두 번째 문장")

    def test_parse_failure_raises_custom_exception_and_logs_repr_preview(self):
        with self.assertLogs(ai_evaluator_service.logger, level="ERROR") as logs:
            with self.assertRaises(CustomException) as raised:
                self._evaluate_text("not-json\ninvalid")

        self.assertEqual(raised.exception.code, ErrorCode.AI_EVALUATION_FAILED)
        self.assertIn("'not-json\\ninvalid'", "\n".join(logs.output))

    def test_missing_required_field_raises_custom_exception(self):
        text = _json_text().replace(
            ',"model_answer":"상황과 성과를 연결한 모범 답변입니다."',
            "",
        )
        with self.assertRaises(CustomException) as raised:
            self._evaluate_text(text)

        self.assertEqual(raised.exception.code, ErrorCode.AI_EVALUATION_FAILED)

    def test_invalid_field_type_raises_custom_exception(self):
        text = _json_text().replace('"S":8', '"S":"high"')
        with self.assertRaises(CustomException) as raised:
            self._evaluate_text(text)

        self.assertEqual(raised.exception.code, ErrorCode.AI_EVALUATION_FAILED)

    def test_string_parsed_value_never_returns_as_success(self):
        response = SimpleNamespace(parsed="invalid", text='"invalid"')
        with patch.object(
            ai_evaluator_service.client.models,
            "generate_content",
            return_value=response,
        ):
            with self.assertRaises(CustomException):
                ai_evaluator_service.evaluate_interview_content("답변", "백엔드")


if __name__ == "__main__":
    unittest.main()
