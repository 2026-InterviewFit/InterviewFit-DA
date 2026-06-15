import json
from google import genai

from app.core.config import settings

client = genai.Client(api_key=settings.GEMINI_API_KEY)


def evaluate_interview_content(script: str, job_category: str):
    try:
        prompt = f"""
        당신은 {job_category} 직무의 전문 면접관입니다. 지원자의 답변을 다음 JSON 형식으로만 엄격하게 출력하세요.
        다른 설명이나 마크다운 형식(```json 등)은 절대 붙이지 마세요.

        필수 JSON 포맷:
        {{
            "star_score": {{ "S": 0, "T": 0, "A": 0, "R": 0 }},
            "feedback": "답변에 대한 상세 코칭 (STAR 구조 중심)",
            "keywords": ["키워드1", "키워드2"],
            "model_answer": "해당 질문에 대한 이상적인 모범 답변 예시"
        }}

        직무: {job_category}
        지원자 답변: {script}
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt
        )

        cleaned_text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned_text)

    except Exception as e:
        print(f"!!! 상세 에러 발생: {e}")
        return {"error": str(e)}
