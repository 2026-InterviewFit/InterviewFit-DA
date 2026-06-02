import json

from app.core.config import settings
from app.core.gemini_client import followup_client
from app.followup.prompts.followup_prompts import create_followup_prompt


def _fallback_follow_up_question(
    job: str,
    original_question: str,
    answer: str,
    interview_style: str = "normal",
) -> str:
    """
    Gemini API 키가 없거나 호출에 실패했을 때 서버 테스트용으로 반환하는 기본 꼬리 질문.
    Spring 연동 테스트가 Gemini 키 때문에 막히지 않도록 둔다.
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
    job: str,
    original_question: str,
    answer: str,
    interview_style: str = "normal",
) -> str:
    """
    지원자의 답변을 기반으로 꼬리 질문 1개 생성.
    Gemini API 사용 가능 시 Gemini 호출, 불가능하면 fallback 질문 반환.
    """

    if followup_client is None:
        return _fallback_follow_up_question(
            job=job,
            original_question=original_question,
            answer=answer,
            interview_style=interview_style,
        )

    prompt = create_followup_prompt(
        job=job,
        original_question=original_question,
        answer=answer,
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
                job=job,
                original_question=original_question,
                answer=answer,
                interview_style=interview_style,
            )

        try:
            result = json.loads(cleaned_text)
            follow_up_question = result.get("follow_up_question")
        except json.JSONDecodeError:
            # 모델이 JSON이 아니라 질문 문장만 반환한 경우도 허용
            follow_up_question = cleaned_text

        if not follow_up_question:
            return _fallback_follow_up_question(
                job=job,
                original_question=original_question,
                answer=answer,
                interview_style=interview_style,
            )

        return str(follow_up_question).strip()

    except Exception as e:
        print(f"Gemini followup error: {e}")

        return _fallback_follow_up_question(
            job=job,
            original_question=original_question,
            answer=answer,
            interview_style=interview_style,
        )