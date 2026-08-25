import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.common.exceptions.custom_exception import CustomException
from app.common.exceptions.error_code import ErrorCode
from app.speech_coaching.services import ai_evaluator_service


VALID_DATA = {
    "star_score": {"S": 8, "T": 7, "A": 9, "R": 8},
    "feedback": "STAR 구조가 명확합니다.",
    "keywords": ["협업", "문제해결"],
    "model_answer": "상황과 성과를 연결한 모범 답변입니다.",
}


def _json_text(feedback: str = "STAR 구조가 명확합니다.") -> str:
    return (
        '{"star_score":{"S":8,"T":7,"A":9,"R":8},'
        f'"feedback":"{feedback}",'
        '"keywords":["협업","문제해결"],'
        '"model_answer":"상황과 성과를 연결한 모범 답변입니다."}'
    )


class AiEvaluatorServiceTest(unittest.TestCase):
    def _evaluate_text(self, text: str):
        response = SimpleNamespace(parsed=None, text=text)
        with patch.object(
            ai_evaluator_service.client.models,
            "generate_content",
            return_value=response,
        ) as generate_content:
            result = ai_evaluator_service.evaluate_interview_content("답변", "백엔드")

        config = generate_content.call_args.kwargs["config"]
        self.assertEqual(config.response_mime_type, "application/json")
        self.assertIs(
            config.response_schema,
            ai_evaluator_service.GeminiAiEvaluation,
        )
        return result

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
