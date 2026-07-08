import json

from app.core.config import settings
from app.core.gemini_client import followup_client
from app.followup.prompts.followup_prompts import create_followup_prompt


def _fallback_follow_up_question(
    job: str | None,
    original_question: str | None,
    answer: str,
    interview_style: str = "normal",
) -> str:
    """
    Gemini API 키가 없거나 호출에 실패했을 때 반환하는 기본 꼬리질문.

    이유:
    - 팀원이 Spring/FE 연결을 테스트할 때 Gemini 키 문제로 전체 흐름이 막히면 안 됨.
    - DB 저장, API 연결, 화면 표시 검증은 fallback 질문만으로도 가능함.
    """

    if interview_style == "pressure":
        return (
            "방금 답변에서 본인의 역할과 판단 근거가 아직 구체적으로 드러나지 않았습니다. "
            "그 상황에서 본인이 직접 결정하고 실행한 부분은 무엇이었나요?"
        )

    if interview_style == "friendly":
        return (
            "좋은 경험입니다. 그 과정에서 배운 점을 이후 비슷한 상황에 어떻게 적용했는지 "
            "조금 더 설명해주실 수 있을까요?"
        )

    return "그 문제를 해결하는 과정에서 가장 어려웠던 판단은 무엇이었고, 왜 그렇게 결정했나요?"


def generate_follow_up_question(
    job: str | None,
    original_question: str | None,
    answer: str,
    interview_style: str = "normal",
) -> str:
    """
    지원자 답변을 기반으로 꼬리질문 1개를 생성한다.

    Spring 쪽에서 interviewId/jobRoleId/questionId를 이용해
    job과 original_question을 가능한 한 채워서 보내지만,
    값이 일부 비어 있어도 FastAPI가 죽지 않도록 기본값을 둔다.
    """

    safe_job = (job or "일반 직무").strip()
    safe_original_question = (original_question or "면접 질문").strip()
    safe_answer = answer.strip()

    if followup_client is None:
        return _fallback_follow_up_question(
            job=safe_job,
            original_question=safe_original_question,
            answer=safe_answer,
            interview_style=interview_style,
        )

    prompt = create_followup_prompt(
        job=safe_job,
        original_question=safe_original_question,
        answer=safe_answer,
        interview_style=interview_style,
    )

    try:
        response = followup_client.models.generate_content(
            model=getattr(settings, "GEMINI_MODEL", "gemini-2.5-flash-lite"),
            contents=prompt,
        )

        cleaned_text = (
            (response.text or "")
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

        if not cleaned_text:
            return _fallback_follow_up_question(
                job=safe_job,
                original_question=safe_original_question,
                answer=safe_answer,
                interview_style=interview_style,
            )

        try:
            result = json.loads(cleaned_text)
            follow_up_question = result.get("follow_up_question")
        except json.JSONDecodeError:
            # 모델이 JSON이 아니라 질문 문장만 반환한 경우도 허용한다.
            follow_up_question = cleaned_text

        if not follow_up_question:
            return _fallback_follow_up_question(
                job=safe_job,
                original_question=safe_original_question,
                answer=safe_answer,
                interview_style=interview_style,
            )

        return str(follow_up_question).strip()

    except Exception as e:
        print(f"Gemini followup error: {e}")

        return _fallback_follow_up_question(
            job=safe_job,
            original_question=safe_original_question,
            answer=safe_answer,
            interview_style=interview_style,
        )