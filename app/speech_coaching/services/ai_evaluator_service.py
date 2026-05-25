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
            print(f"선택된 모델: {m.name}")
            return genai.GenerativeModel(m.name)
    # 실패 시 기본 모델 반환
    return genai.GenerativeModel('gemini-1.5-flash')


def evaluate_interview_content(script: str, job_category: str):
    try:
        model = get_best_model()  # 자동 모델 선택 호출

        prompt = f"""
        당신은 전문 면접관입니다. 지원자의 답변을 분석하세요.
        직무: {job_category}
        답변: {script}

        STAR 기법(상황, 과제, 행동, 결과) 구조 분석과 직무 키워드 포함 여부를 JSON 형식으로 출력하세요.
        반드시 아래 포맷만 출력하세요:
        {{"star_score": {{"S": 0, "T": 0, "A": 0, "R": 0}}, "feedback": "...", "keywords": []}}
        """

        response = model.generate_content(prompt)
        cleaned_text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned_text)

    except Exception as e:
        print(f"!!! 상세 에러 발생: {e}")
        return {"error": str(e)}