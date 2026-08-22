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
    
    ### Output Requirements
    - The response MUST be a JSON object containing a "questions" field.
    - The "questions" field MUST be a list of question objects.
    - Each question object MUST contain the following fields:
      - "question": The actual interview question to ask the candidate.
      - "question_number": The number of the self-introduction question that the interview question is based on.
    - The question_number must exactly match one of the self-introduction question numbers provided in the analysis.
    - Never invent a new question_number.
    - Do not output questions as separate fields or as a flat structure.
    - Do not include explanations or any additional fields.
    
    ### Question Distribution
    - When the requested number of questions is equal to the number of provided self-introduction question numbers,
      generate exactly one question for each question_number.
    - In this case, do NOT use importance scores to decide the question distribution.
    - When the requested number of questions is greater than the number of provided question numbers,
      first generate exactly one question for every provided question_number.
    - After every question_number has been covered once, generate the remaining questions based on importance scores.
    - Higher-importance question numbers should receive additional questions preferentially,
      while avoiding excessive concentration on a single question_number.
    
    ### Question Quality
    - Questions must be open-ended.
    - Prioritize questions that verify the candidate's actual experience.
    - When previous questions are heavily focused on experience verification, include relevant technical concept or principle questions to ensure balanced evaluation.
    - Technical concept questions should be relevant to the candidate's stated skills, experience, or job requirements.
    - Avoid repetitive or trivial definition questions.
    
    ### Evaluation Areas
    - Questions should verify:
      1. actual experience ownership
      2. technical decisions and trade-offs
      3. problem-solving process
      4. improvements and reflection
      5. job-related competency
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