def create_analysis_prompt(coverletter: str):
    system_prompt = """
You are an experienced recruiter and interview expert with over 10 years of professional experience.
Analyze the applicant's cover letter and generate structured data that can be used for interview preparation.

Rules:
- Analyze each cover letter question separately.
- Extract the question and answer exactly as written without modifying or summarizing them.
- Base your analysis only on the information provided in the cover letter.
- Do not infer or fabricate information that is not explicitly stated.
- Write all responses in Korean.
"""

    user_prompt = f"""
다음 자기소개서를 분석하세요.

[자기소개서]
{coverletter}


각 문항별로 아래 정보를 생성하세요.

- question_number:
  문항 번호

- question:
  자기소개서 질문 원문

- answer:
  해당 질문의 답변 원문

- summary:
  답변 요약

- star:
  Situation: 경험 배경
  Task: 해결해야 할 문제 또는 목표
  Action: 수행한 행동
  Result: 결과 및 성과

- skills:
  답변에서 확인되는 기술 및 역량

- strengths:
  답변의 강점

- improvements:
  개선하면 좋은 점

- importance_score:
  면접 준비 중요도 (0~100)


추가 분석:

1. 자기소개서 전체를 기준으로 가장 적합한 직무를 하나 선택하세요.

가능한 직무:
- 프론트엔드
- 백엔드
- AI
- 데이터분석
- 마케팅
- 인사
- 재무·회계
- UX/UI 디자이너
- 영상 제작
- 그래픽 디자이너
- 기타

2. 직무가 목록에 있으면:
- job_name: 해당 직무명
- custom_job_name: null

3. 목록에 없으면:
- job_name: 기타
- custom_job_name: 실제 추천 직무명

4. score:
추천 직무와 자기소개서 적합도를 0~100으로 평가

5. job_reason:
추천 이유 작성
"""

    return system_prompt, user_prompt