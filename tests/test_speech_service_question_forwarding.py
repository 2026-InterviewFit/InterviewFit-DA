import os
import tempfile
import unittest
from unittest.mock import patch

from app.speech_coaching.schemas.speech_schema import ExternalAiAnalysis
from app.speech_coaching.services import speech_service


class SpeechServiceQuestionForwardingTest(unittest.TestCase):
    def test_original_question_is_forwarded_without_changing_response_shape(self):
        question = "입사 후 원하지 않는 부서에 배치된다면 어떻게 하겠습니까?"
        ai_result = {
            "star_score": {"S": 5, "T": 5, "A": 5, "R": 5},
            "feedback": "실제 답변에 대한 개선 피드백입니다.",
            "keywords": ["적응", "성장"],
            "model_answer": "저는 우선 회사의 배치 이유를 이해하고 업무에 적응하겠습니다.",
        }
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as video:
            video_path = video.name

        try:
            with (
                patch.object(
                    speech_service,
                    "extract_audio_and_get_duration",
                    return_value=(True, 10.0),
                ),
                patch.object(
                    speech_service,
                    "analyze_audio_metrics",
                    return_value={
                        "total_silence_seconds": 1.0,
                        "pause_rate_percent": 10.0,
                        "critical_silence_count": 0,
                    },
                ),
                patch.object(
                    speech_service,
                    "run_free_whisper_stt",
                    return_value="사용자 답변",
                ),
                patch.object(
                    speech_service.external_ai_service,
                    "analyze_interview_answer",
                    return_value=None,
                ) as external_analyze,
                patch.object(
                    speech_service.ai_evaluator_service,
                    "evaluate_interview_content",
                    return_value=ai_result,
                ) as evaluate,
            ):
                response = speech_service.process_interview_analysis(
                    source_type="local",
                    source_value=video_path,
                    job_category="백엔드",
                    interview_id=1,
                    original_question=question,
                )

            external_analyze.assert_called_once_with(
                question=question,
                answer="사용자 답변",
            )
            evaluate.assert_called_once_with("사용자 답변", "백엔드", question, None)
            body = response.model_dump()
            self.assertEqual(body["original_question"], question)
            self.assertIsNone(body["external_ai_analysis"])
            self.assertEqual(body["ai_evaluation"]["model_answer"], ai_result["model_answer"])
            self.assertEqual(
                set(body["ai_evaluation"]),
                {"star_score", "feedback", "keywords", "model_answer"},
            )
        finally:
            os.unlink(video_path)

    def test_external_ai_result_is_forwarded_to_gemini_and_response(self):
        question = "문제를 해결한 경험을 말씀해 주세요."
        external_analysis = ExternalAiAnalysis.model_validate({
            "star": {
                "situation": "장애 상황을 설명했습니다.",
                "task": "복구 목표를 설명했습니다.",
                "action": "원인을 찾아 수정했습니다.",
                "result": "서비스를 정상화했습니다.",
            },
            "strengths": ["행동이 구체적입니다."],
            "improvements": ["성과를 수치화해야 합니다."],
        })
        ai_result = {
            "star_score": {"S": 7, "T": 7, "A": 8, "R": 6},
            "feedback": "구체적인 성과를 보완하세요.",
            "keywords": ["장애 대응"],
            "model_answer": "장애 원인을 분석하고 복구한 경험이 있습니다.",
        }
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as video:
            video_path = video.name

        try:
            with (
                patch.object(speech_service, "extract_audio_and_get_duration", return_value=(True, 10.0)),
                patch.object(
                    speech_service,
                    "analyze_audio_metrics",
                    return_value={
                        "total_silence_seconds": 1.0,
                        "pause_rate_percent": 10.0,
                        "critical_silence_count": 0,
                    },
                ),
                patch.object(speech_service, "run_free_whisper_stt", return_value="사용자 답변"),
                patch.object(
                    speech_service.external_ai_service,
                    "analyze_interview_answer",
                    return_value=external_analysis,
                ),
                patch.object(
                    speech_service.ai_evaluator_service,
                    "evaluate_interview_content",
                    return_value=ai_result,
                ) as evaluate,
            ):
                response = speech_service.process_interview_analysis(
                    source_type="local",
                    source_value=video_path,
                    job_category="백엔드",
                    interview_id=1,
                    original_question=question,
                )

            evaluate.assert_called_once_with(
                "사용자 답변",
                "백엔드",
                question,
                external_analysis,
            )
            self.assertEqual(
                response.external_ai_analysis.model_dump(),
                external_analysis.model_dump(),
            )
        finally:
            os.unlink(video_path)


class SpeechMetricsTest(unittest.TestCase):
    def test_filler_words_are_counted_regardless_of_punctuation(self):
        metrics = speech_service.analyze_speech_metrics(
            "음, 저는 어... 그! 이제 시작하겠습니다. 음?",
            duration_seconds=10.0,
        )

        self.assertEqual(
            metrics["filler_word_analysis"],
            {
                "details": {
                    "음": 2,
                    "어": 1,
                    "그": 1,
                    "이제": 1,
                    "약간": 0,
                    "저": 0,
                },
                "total_count": 5,
            },
        )

    def test_consecutive_repeats_are_detected_across_punctuation(self):
        repeats = speech_service.find_consecutive_repeats(
            "프로젝트, 프로젝트... 프로젝트를 진행했습니다. 네! 네?",
        )

        self.assertEqual(
            repeats,
            [
                {"expression": "프로젝트", "count": 2},
                {"expression": "네", "count": 2},
            ],
        )


if __name__ == "__main__":
    unittest.main()
