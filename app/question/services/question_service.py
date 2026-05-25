import json

from app.core.gemini_client import question_client
from app.question.prompts.question_prompts import create_question_prompt

def generate_first_question(count: int, job: str, resume_content: str | None = None):
    prompt = create_question_prompt(count, job, resume_content)

    response = question_client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt
    )

    question_list = json.loads(response.text)

    return question_list