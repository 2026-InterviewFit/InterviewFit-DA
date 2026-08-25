import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pydantic import ValidationError

from app.common.exceptions.custom_exception import CustomException
from app.common.exceptions.error_code import ErrorCode
from app.group_interview.prompts.comparison_prompts import (
    SYSTEM_PROMPT,
    create_comparison_prompt,
)
from app.group_interview.schemas.comparison_schema import (
    GroupComparisonFeedbackRequest,
    GroupComparisonFeedbackResponse,
)
from app.group_interview.services import comparison_service


def participant(
    participant_id: int,
    *,
    content: float = 80,
    speech: float = 80,
    gaze: float = 80,
    pose: float = 80,
    blink: float = 80,
    total: float = 80,
    rank: int = 1,
) -> dict:
    return {
        "participantUserId": participant_id,
        "contentScore": content,
        "speechScore": speech,
        "gazeScore": gaze,
        "poseScore": pose,
        "blinkScore": blink,
        "totalScore": total,
        "rank": rank,
    }


def request_for(items: list[dict], question_index: int | None = 0):
    return GroupComparisonFeedbackRequest.model_validate(
        {"roomId": 1, "questionIndex": question_index, "participants": items}
    )


def feedback(participant_id: int) -> dict:
    return {
        "participantUserId": participant_id,
        "strengthSummary": "답변 내용의 구체성이 상대적인 강점으로 나타났습니다.",
        "improvementSummary": "핵심 내용을 간결하게 전달하는 연습이 도움이 됩니다.",
    }


class GroupComparisonRequestValidationTest(unittest.TestCase):
    def test_two_participant_question_request(self):
        request = request_for([participant(3), participant(4, rank=2)])
        self.assertEqual(len(request.participants), 2)
        self.assertEqual(request.questionIndex, 0)

    def test_four_participant_question_request(self):
        request = request_for(
            [participant(index, rank=index) for index in range(1, 5)],
            question_index=2,
        )
        self.assertEqual(len(request.participants), 4)

    def test_overall_request(self):
        request = request_for([participant(3), participant(4)], question_index=None)
        self.assertIsNone(request.questionIndex)
        prompt = create_comparison_prompt(request)
        self.assertIn("전체 면접 결과에서는", prompt)
        self.assertIn("2~3문장", prompt)
        self.assertIn("반복 연습", prompt)

    def test_tied_participants_are_accepted(self):
        request = request_for([participant(3), participant(4)])
        self.assertEqual(request.participants[0].totalScore, request.participants[1].totalScore)

    def test_similar_scores_are_accepted(self):
        request = request_for(
            [participant(3), participant(4, content=79, speech=81, total=79.5)]
        )
        self.assertEqual(len(request.participants), 2)

    def test_clear_strength_is_preserved_without_recalculation(self):
        request = request_for(
            [participant(3, content=100, total=85), participant(4, content=50, total=70)]
        )
        self.assertEqual(request.participants[0].contentScore, 100)
        self.assertEqual(request.participants[0].totalScore, 85)

    def test_one_participant_is_rejected(self):
        with self.assertRaises(ValidationError):
            request_for([participant(3)])

    def test_duplicate_participant_id_is_rejected(self):
        with self.assertRaises(ValidationError):
            request_for([participant(3), participant(3)])

    def test_negative_score_is_rejected(self):
        with self.assertRaises(ValidationError):
            request_for([participant(3, content=-1), participant(4)])

    def test_score_over_100_is_rejected(self):
        with self.assertRaises(ValidationError):
            request_for([participant(3, gaze=101), participant(4)])

    def test_invalid_question_index_is_rejected(self):
        for invalid_index in (-1, 3):
            with self.subTest(question_index=invalid_index):
                with self.assertRaises(ValidationError):
                    request_for(
                        [participant(3), participant(4)],
                        question_index=invalid_index,
                    )


class GroupComparisonServiceTest(unittest.TestCase):
    def setUp(self):
        self.request = request_for([participant(3), participant(4, rank=2)])

    def _generate(self, response_participants: list[dict]):
        parsed = GroupComparisonFeedbackResponse.model_validate(
            {"participants": response_participants}
        )
        generate_content = Mock(return_value=SimpleNamespace(parsed=parsed))
        client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
        with patch.object(comparison_service, "comparison_client", client):
            result = comparison_service.generate_comparison_feedback(self.request)
        return result, generate_content

    def test_successful_structured_output_and_contract(self):
        result, generate_content = self._generate([feedback(3), feedback(4)])

        self.assertEqual(
            result.model_dump(),
            {"participants": [feedback(3), feedback(4)]},
        )
        config = generate_content.call_args.kwargs["config"]
        self.assertEqual(config.response_mime_type, "application/json")
        self.assertIs(config.response_schema, GroupComparisonFeedbackResponse)

    def test_schema_has_no_additional_properties(self):
        schema = GroupComparisonFeedbackResponse.model_json_schema()

        def contains_additional_properties(value):
            if isinstance(value, dict):
                return "additionalProperties" in value or any(
                    contains_additional_properties(child) for child in value.values()
                )
            if isinstance(value, list):
                return any(contains_additional_properties(child) for child in value)
            return False

        self.assertFalse(contains_additional_properties(schema))

    def test_gemini_call_failure_raises_custom_exception(self):
        generate_content = Mock(side_effect=RuntimeError("Gemini unavailable"))
        client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
        with patch.object(comparison_service, "comparison_client", client):
            with self.assertRaises(CustomException) as raised:
                comparison_service.generate_comparison_feedback(self.request)

        self.assertEqual(
            raised.exception.code,
            ErrorCode.GROUP_COMPARISON_FEEDBACK_FAILED,
        )

    def test_missing_participant_is_rejected(self):
        with self.assertRaises(CustomException):
            self._generate([feedback(3)])

    def test_unrequested_participant_is_rejected(self):
        with self.assertRaises(CustomException):
            self._generate([feedback(3), feedback(4), feedback(5)])

    def test_duplicate_response_participant_is_rejected(self):
        with self.assertRaises(CustomException):
            self._generate([feedback(3), feedback(3)])

    def test_blank_or_missing_summary_is_rejected(self):
        for invalid_feedback in (
            {
                "participantUserId": 3,
                "improvementSummary": "개선 문장",
            },
            {
                "participantUserId": 3,
                "strengthSummary": "강점 문장",
                "improvementSummary": "   ",
            },
        ):
            with self.subTest(feedback=invalid_feedback):
                with self.assertRaises(ValidationError):
                    GroupComparisonFeedbackResponse.model_validate(
                        {"participants": [invalid_feedback, feedback(4)]}
                    )

    def test_prompt_contains_required_safety_policies(self):
        for phrase in (
            "totalScore와 rank를 변경하지 마세요",
            "가중치를 다시 적용하지 마세요",
            "합격/불합격",
            "동점 또는 유사한 점수",
            "성격, 능력, 정직성, 자신감",
        ):
            self.assertIn(phrase, SYSTEM_PROMPT)
        self.assertIn("이번 질문에서는", create_comparison_prompt(self.request))


if __name__ == "__main__":
    unittest.main()
