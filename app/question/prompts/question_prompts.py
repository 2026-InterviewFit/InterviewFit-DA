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

    Follow these rules:
    - Output exactly ONE JSON array.
    - Each element in the array must be a single Korean interview question.
    - One string represents exactly one question.
    - Do NOT combine multiple questions into one string.
    - Do NOT split one question into multiple strings.
    - The number of elements in the array MUST exactly match the requested question count.
    - Do NOT wrap the array in an object.
    - Do NOT include markdown, code blocks, explanations, or extra text.
    - Do NOT generate duplicate or similar questions.
    
    Example:
    [질문1", "질문2", "질문3"]
    """)

    # USER CONTEXT (실제 작업 내용)
    if cover_letter:
        additional_rule = f"""
    지원자의 자기소개서:
    {cover_letter}

    질문 생성 기준:
    - 자기소개서의 경험과 내용을 기반으로 질문을 생성한다.
    - 지원자의 경험, 기술, 성과를 검증할 수 있는 질문을 포함한다.
    - 단순 사실 확인보다 문제 해결 과정과 의사결정 과정을 확인하는 질문을 우선한다.
    - 프로젝트 경험이 있다면 기술 선택 이유와 개선 경험을 확인하는 질문을 포함한다.
    - 지원자가 자신의 경험을 구체적으로 설명할 수 있는 질문을 생성한다.
    """
    else:
        additional_rule = """
    자기소개서가 제공되지 않았다.

    질문 생성 기준:
    - 지원 직무 역량을 평가할 수 있는 질문을 생성한다.
    - 아래 영역을 균형 있게 고려한다.

    1. 직무 기술 역량
    2. 프로젝트 경험
    3. 문제 해결 능력
    4. 협업 능력
    5. 직무 이해도
    6. 성과 및 개선 경험
    7. 의사결정 능력
    8. 성장 가능성

    - 특정 영역에만 치우치지 않는다.
    """

    user_prompt = dedent(f"""
    직무: {job}
    생성할 질문 개수: {count}
    {additional_rule}
    출력 규칙:
    - 반드시 {count}개의 질문만 생성한다.
    - JSON 배열의 각 항목은 하나의 독립적인 면접 질문이어야 한다.
    - 배열 안에 질문 개수 외의 다른 텍스트를 포함하지 않는다.
    
    이제 질문을 생성하라.
    """).strip()

    return system_prompt, user_prompt