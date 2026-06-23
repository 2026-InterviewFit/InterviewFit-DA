from textwrap import dedent

def create_question_prompt(count: int, job: str, cover_letter: str | None = None):
    """
    returns:
        system_prompt: str
        user_prompt: str
    """

    # SYSTEM PROMPT (절대 규칙)
    system_prompt = dedent("""
    You are a strict JSON generator.

    You MUST follow these rules:
    - Output ONLY valid JSON.
    - Do NOT include markdown (```), explanations, or extra text.
    - Output must be a JSON array of strings.
    - Each string must be a single interview question.
    - No exceptions.
    
    """)

    # USER CONTEXT (실제 작업 내용)
    if cover_letter:
        additional_rule = f"""
    지원자의 자기소개서 내용:
    {cover_letter}
    
    면접 질문 생성 기준:
    - 자기소개서 내용을 기반으로 질문을 생성하라.
    - 경험, 기술, 성과를 깊이 검증하는 질문을 포함하라.
    - 문제 해결 과정과 의사결정 과정에 집중하라.
    - 자기소개서 기반 후속 질문을 포함하라.
    """
    else:
        additional_rule = """
    자기소개서가 제공되지 않았다.
    
    면접 질문 생성 기준:
    - 직무 중심 역량 평가 질문을 생성하라.
    - 다음 영역에서 균형 있게 선택하라:
      1. 직무 기술 역량
      2. 프로젝트 경험
      3. 문제 해결 능력
      4. 협업 능력
      5. 직무 이해도
      6. 성과 및 개선 경험
      7. 의사결정 능력
      8. 성장 가능성
    - 중복 질문 금지
    - 다양한 관점 유지
    """

    user_prompt = dedent(f"""
    직무: {job}
    질문 개수: {count}
    {additional_rule}
    OUTPUT FORMAT (VERY IMPORTANT):
    Return a JSON array of exactly {count} strings.
    
    Example:
    ["질문1", "질문2", "질문3"]
    
    Now generate questions.
    """).strip()

    return system_prompt, user_prompt