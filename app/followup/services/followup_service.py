import json

from app.core.gemini_client import followup_client
from app.followup.prompts.followup_prompts import create_followup_prompt


def generate_follow_up_question(
    job: str,
    original_question: str,
    answer: str,
    interview_style: str = "normal",
) -> str:
    prompt = create_followup_prompt(
        job=job,
        original_question=original_question,
        answer=answer,
        interview_style=interview_style,
    )

    response = followup_client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
    )

    cleaned_text = (
        response.text
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    try:
        result = json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Gemini 응답을 JSON으로 파싱할 수 없습니다. 응답 내용: {cleaned_text}"
        ) from e

    if "follow_up_question" not in result:
        raise ValueError(
            f"Gemini 응답에 follow_up_question 필드가 없습니다. 응답 내용: {result}"
        )

    return result["follow_up_question"]