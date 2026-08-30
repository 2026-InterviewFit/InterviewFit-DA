import unittest
from unittest.mock import patch

import httpx

from app.speech_coaching.services import external_ai_service


VALID_RESPONSE = {
    "star": {
        "situation": "문제 상황을 설명했습니다.",
        "task": "담당 목표를 제시했습니다.",
        "action": "직접 수행한 행동을 설명했습니다.",
        "result": "성과를 수치로 제시했습니다.",
    },
    "strengths": ["행동이 구체적입니다.", "성과가 명확합니다."],
    "improvements": ["상황 설명을 더 간결하게 해야 합니다."],
}


def _response(status_code: int = 200, **kwargs) -> httpx.Response:
    return httpx.Response(
        status_code,
        request=httpx.Request("POST", "http://external-ai/analyze"),
        **kwargs,
    )


class ExternalAiServiceTest(unittest.TestCase):
    def test_successful_response_is_validated_and_returned(self):
        with (
            patch.object(external_ai_service.settings, "AI_SERVER_URL", "http://external-ai/"),
            patch.object(external_ai_service.httpx, "post", return_value=_response(json=VALID_RESPONSE)) as post,
        ):
            result = external_ai_service.analyze_interview_answer("질문", "답변")

        self.assertIsNotNone(result)
        self.assertEqual(result.model_dump(), VALID_RESPONSE)
        self.assertEqual(post.call_args.args[0], "http://external-ai/analyze")
        self.assertEqual(
            post.call_args.kwargs["json"],
            {"question": "질문", "answer": "답변"},
        )
        timeout = post.call_args.kwargs["timeout"]
        self.assertEqual(timeout.connect, external_ai_service.settings.AI_SERVER_CONNECT_TIMEOUT_SECONDS)
        self.assertEqual(timeout.read, external_ai_service.settings.AI_SERVER_READ_TIMEOUT_SECONDS)

    def test_timeout_returns_none_without_logging_question_or_answer(self):
        question = "로그에 남으면 안 되는 질문"
        answer = "로그에 남으면 안 되는 답변"
        timeout = httpx.ReadTimeout(
            "timed out",
            request=httpx.Request("POST", "http://external-ai/analyze"),
        )
        with (
            patch.object(external_ai_service.settings, "AI_SERVER_URL", "http://external-ai"),
            patch.object(external_ai_service.httpx, "post", side_effect=timeout),
            self.assertLogs(external_ai_service.logger, level="WARNING") as logs,
        ):
            result = external_ai_service.analyze_interview_answer(question, answer)

        self.assertIsNone(result)
        output = "\n".join(logs.output)
        self.assertNotIn(question, output)
        self.assertNotIn(answer, output)

    def test_http_error_status_returns_none(self):
        for status_code in (400, 500):
            with self.subTest(status_code=status_code):
                with (
                    patch.object(external_ai_service.settings, "AI_SERVER_URL", "http://external-ai"),
                    patch.object(
                        external_ai_service.httpx,
                        "post",
                        return_value=_response(status_code, json={"detail": "failed"}),
                    ),
                ):
                    result = external_ai_service.analyze_interview_answer("질문", "답변")

                self.assertIsNone(result)

    def test_connection_failure_returns_none(self):
        error = httpx.ConnectError(
            "connection failed",
            request=httpx.Request("POST", "http://external-ai/analyze"),
        )
        with (
            patch.object(external_ai_service.settings, "AI_SERVER_URL", "http://external-ai"),
            patch.object(external_ai_service.httpx, "post", side_effect=error),
        ):
            result = external_ai_service.analyze_interview_answer("질문", "답변")

        self.assertIsNone(result)

    def test_invalid_json_returns_none(self):
        with (
            patch.object(external_ai_service.settings, "AI_SERVER_URL", "http://external-ai"),
            patch.object(
                external_ai_service.httpx,
                "post",
                return_value=_response(content=b"not-json"),
            ),
        ):
            result = external_ai_service.analyze_interview_answer("질문", "답변")

        self.assertIsNone(result)

    def test_schema_error_returns_none(self):
        invalid_response = {"star": {"situation": "상황"}, "strengths": []}
        with (
            patch.object(external_ai_service.settings, "AI_SERVER_URL", "http://external-ai"),
            patch.object(
                external_ai_service.httpx,
                "post",
                return_value=_response(json=invalid_response),
            ),
        ):
            result = external_ai_service.analyze_interview_answer("질문", "답변")

        self.assertIsNone(result)

    def test_unset_url_skips_request(self):
        with (
            patch.object(external_ai_service.settings, "AI_SERVER_URL", "  "),
            patch.object(external_ai_service.httpx, "post") as post,
        ):
            result = external_ai_service.analyze_interview_answer("질문", "답변")

        self.assertIsNone(result)
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
