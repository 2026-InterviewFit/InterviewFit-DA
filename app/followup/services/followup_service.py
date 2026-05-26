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

    # 모델이 가끔 코드블럭을 붙이는 경우 방어
    cleaned_text = (
        response.text
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    result = json.loads(cleaned_text)
    return result["follow_up_question"]