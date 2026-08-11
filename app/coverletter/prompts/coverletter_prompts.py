from app.coverletter.schemas.coverletter_schema import CoverLetterQuestionItem

def create_coverletter_extraction_prompt(coverletter: str):
    system_prompt = """
You are an information extraction expert.
Your task is to extract each cover letter question and answer exactly as written.

Rules:
- Extract each question separately.
- Preserve the original wording.
- Do not modify, summarize, or analyze the content.
- Do not infer missing information.
- Write all responses in Korean.
"""

    user_prompt = f"""
다음 자기소개서를 문항별로 분리하세요.

[자기소개서]
{coverletter}

각 문항마다 아래 항목을 생성하세요.

- question_number:
  문항 번호

- question:
  자기소개서 질문 원문

- answer:
  해당 질문의 답변 원문

- summary:
  답변을 2~3문장으로 간결하게 요약
"""

    return system_prompt, user_prompt

def create_coverletter_importance_prompt(questions: list[CoverLetterQuestionItem]):
    system_prompt = """
You are an experienced recruiter and interview expert.
Your task is to evaluate the interview preparation importance of each cover letter question.

Rules:
- Compare all questions with each other.
- Assign relative importance scores.
- Write all responses in Korean.
- Use a concise and objective "~함", "~임" style.
- Do not use polite "~합니다" style.

- Consider:
  - relevance to job competency
  - possibility of follow-up interview questions
  - depth of experience
  - STAR structure potential
  - ability to evaluate candidate capability
  
- Score range: 0.0~100.0
"""

    question_text = "\n\n".join(
        [
            f"""
[문항 {q.question_number}]
질문:
{q.question}
답변:
{q.answer}
"""
            for q in questions
        ]
    )

    user_prompt = f"""
다음 자기소개서 문항들의 면접 준비 중요도를 평가하세요.

{question_text}

각 문항마다
- question_number
- importance_score
를 생성하세요.
"""

    return system_prompt, user_prompt

def create_coverletter_analysis_prompt(question: str, answer: str, summary: str):
    system_prompt = """
You are an experienced recruiter and interview expert.
Analyze a single cover letter answer.

Rules:
- Analyze only the provided information.
- Do not infer missing experiences.
- If STAR information is missing, leave it empty.
- Write all responses in Korean.
- Use a concise and objective "~함", "~임" style.
- Do not use polite "~합니다" style.
"""

    user_prompt = f"""
다음 자기소개서 문항을 분석하세요.
[질문]
{question}
[답변]
{answer}
[요약]
{summary}

생성 항목:

- star
  - situation: 경험 배경
  - task: 해결해야 할 문제 또는 목표
  - action: 수행한 행동
  - result: 결과 및 성과

- skills
  답변에서 확인되는 기술 및 역량

- strengths
  답변의 강점

- improvements
  개선하면 좋은 점
"""

    return system_prompt, user_prompt

def create_job_recommendation_prompt(coverletter: str):
    system_prompt = """
You are a professional career consultant.
Recommend the most appropriate job based on the applicant's entire cover letter.

Rules:
- Consider the entire cover letter.
- Select only one job.
- Base your recommendation only on the provided content.
- Do not invent experiences or skills.
- Write all responses in Korean.
- Use a concise and objective "~함", "~임" style.
- Do not use polite "~합니다" style.
"""

    user_prompt = f"""
다음 자기소개서를 보고 가장 적합한 직무를 추천하세요.

[자기소개서]
{coverletter}

가능한 직무는 다음과 같습니다.
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

생성 항목:

- job_name
  위 목록 중 하나를 선택

- custom_job_name
  job_name이 "기타"인 경우 실제 직무명을 작성하고,
  그렇지 않으면 null

- score
  자기소개서와 직무의 적합도 (0.0~100.0)

- job_reason
  추천 이유를 작성
"""

    return system_prompt, user_prompt