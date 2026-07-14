from textwrap import dedent

def create_question_prompt(count: int, job: str, cover_letter: str, previous_questions: list[str] | None = None) -> tuple[str, str]:
    """
    returns:
        system_prompt: str
        user_prompt: str
    """

    # SYSTEM PROMPT (절대 규칙)
    system_prompt = dedent("""
    You generate Korean interview questions as strict JSON.

    Rules:
    - Output only one JSON array.
    - Each element must be exactly one interview question string.
    - The number of questions must exactly match the requested count.
    - Do not include markdown, explanations, or any extra text.
    - Do not generate duplicate questions.
    - Do not generate questions with the same evaluation purpose.
    - Questions must be open-ended interview questions.
    - Avoid simple fact-checking questions.
    - When generating multiple questions, evaluate different competencies.
    """)

    # USER CONTEXT (실제 작업 내용)
    additional_rule = f"""
    ### 지원자의 자기소개서
    {cover_letter}

    ### 생성 기준
    - 자기소개서의 경험과 내용을 기반으로 질문을 생성한다.
    - 지원자의 경험, 기술 선택, 성과, 문제 해결 과정, 의사결정 과정을 확인한다.
    - 단순 사실 확인보다 행동 과정과 판단 근거를 확인하는 질문을 우선한다.
    """

    previous_rule = ""
    if previous_questions:
        previous_rule = f"""
    ### 기존 질문 목록
    {chr(10).join(f"- {q}" for q in previous_questions)}

    ### 기존 질문 제외 규칙
    - 기존 질문과 동일한 질문을 생성하지 않는다.
    - 표현만 변경한 유사 질문을 생성하지 않는다.
    - 기존 질문과 동일한 경험이나 역량을 평가하는 질문을 피한다.
    """

    user_prompt = dedent(f"""
    직무: {job}
    생성할 질문 개수: {count}
    {additional_rule}
    {previous_rule}
    ### 출력 규칙
    - 반드시 {count}개의 질문만 생성한다.
    - JSON 배열 형태로만 반환한다.
    - 각 배열 요소는 하나의 독립적인 면접 질문이어야 한다.
    질문을 생성하라.
    """).strip()

    return system_prompt, user_prompt