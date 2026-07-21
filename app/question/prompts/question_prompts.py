from textwrap import dedent

from app.question.schemas.question_schema import CoverLetterQuestionAnalysisDto2, StarDto

def create_question_prompt(
    count: int,
    job: str,
    analyses: list[CoverLetterQuestionAnalysisDto2],
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
    - Output only one JSON array.
    - Generate exactly the requested number of questions.
    - Each element must contain one interview question.
    - Questions must be open-ended.
    - Prioritize experiences with higher importance scores.
    - Focus more questions on high-importance experiences.
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
    - 각 질문은 서로 다른 평가 목적을 가져야 합니다.
    - 반드시 JSON 배열만 반환하세요.
    """)

    return system_prompt, user_prompt

def format_star(star: StarDto | None) -> str:
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