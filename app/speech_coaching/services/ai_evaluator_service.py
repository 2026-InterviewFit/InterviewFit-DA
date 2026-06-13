import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def get_best_model():
    """사용 가능한 모델 중 flash 모델을 찾아 반환"""
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods and 'flash' in m.name:
            #print(f"선택된 모델: {m.name}")
            return genai.GenerativeModel(m.name)
    # 실패 시 기본 모델 반환
    return genai.GenerativeModel('gemini-1.5-flash')


def evaluate_interview_content(script: str, job_category: str):
    try:
        model = get_best_model()  # 자동 모델 선택 호출

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

        response = model.generate_content(prompt)
        cleaned_text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned_text)

    except Exception as e:
        print(f"!!! 상세 에러 발생: {e}")
        return {"error": str(e)}