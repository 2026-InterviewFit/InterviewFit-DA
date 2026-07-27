from textwrap import dedent

from app.question.schemas.question_schema import CoverLetterQuestionAnalysisItem, StarItem

def create_question_prompt(
    count: int,
    job: str,
    analyses: list[CoverLetterQuestionAnalysisItem],
    previous_questions: list[str] | None
) -> tuple[str, str]:
    """
    returns:
        system_prompt: str
        user_prompt: str
    """

    system_prompt = dedent("""
    You are an expert Korean interview coach.
    Generate personalized interview questions based on the candidate's self-introduction analysis.

    Rules:
    - Output only a JSON object that conforms to the provided response schema.
    - Generate exactly the requested number of questions.
    - Each question must include:
      - "question"
      - "question_number"
    - The question_number must exactly match one of the self-introduction question numbers provided in the analysis.
    - Never invent a new question_number.
    - Questions must be open-ended.
    - Prioritize experiences with higher importance scores.
    - However, ensure balanced coverage across all provided self-introduction question numbers.
    - Do not generate all questions from only the highest-scored experiences.
    - Generate at least one question for each question_number before assigning additional questions.
    - Do not ask questions that only test simple definitions.
    - Questions should verify:
      1. actual experience ownership
      2. technical decisions and trade-offs
      3. problem-solving process
      4. improvements and reflection
      5. job-related competency
    - Avoid duplicate or similar questions.
    - Do not generate explanations or additional text.
    """)

    analysis_text = "\n\n".join(f"""
    [문항 {analysis.question_number}]
    중요도 점수: {analysis.importance_score}
    자기소개서 요약:
    {analysis.summary}
    STAR:
    {format_star(analysis.star)}
    강점:
    {", ".join(analysis.strengths) if analysis.strengths else "-"}
    개선점:
    {", ".join(analysis.improvements) if analysis.improvements else "-"}
    기술 스택:
    {", ".join(analysis.skills) if analysis.skills else "-"}
    핵심 키워드:
    {", ".join(analysis.keywords) if analysis.keywords else "-"}
    """
    for analysis in analyses
    )

    previous_rule = ""
    if previous_questions:
        question_text = "\n".join(f"""
    - {q}
    """ for q in previous_questions)
        previous_rule = f"""
    ### 이미 생성된 질문
    {question_text}
    위 질문과 동일하거나 평가 목적이 유사한 질문은 생성하지 마세요.
    """

    user_prompt = dedent(f"""
    ### 지원 직무
    {job}
    ### 생성할 질문 개수
    {count}개
    ### 자기소개서 분석 결과
    {analysis_text}
    
    {previous_rule}
    질문 목적:
    1. 지원자가 실제로 해당 경험을 수행했는지 검증
    2. 기술 선택의 이유와 trade-off 확인
    3. 문제 발생 상황과 해결 과정 확인
    4. 개선 방향과 회고 확인
    5. 지원자의 역량과 직무 적합성 확인
    조건:
    - 반드시 {count}개의 질문만 생성하세요.
    - 중요도가 높은 경험을 우선 활용하세요.
    - 전체 질문 개수가 문항 개수보다 많다면, 중요도 점수가 높은 문항에 추가 질문을 배정하세요.
    - 특정 하나의 문항에 질문이 집중되지 않도록 하세요.
    - 각 질문은 서로 다른 평가 목적을 가져야 합니다.
    """)

    return system_prompt, user_prompt

def format_star(star: StarItem | None) -> str:
    if not star:
        return "-"

    return f"""
    Situation:
    {star.situation}
    Task:
    {star.task}
    Action:
    {star.action}
    Result:
    {star.result}
    """