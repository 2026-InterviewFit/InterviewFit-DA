def create_question_prompt(count: int, job: str, cover_letter: str | None = None):
    cover_letter_prompt = ""

    if cover_letter:
        cover_letter_prompt = f"""
            지원자의 자기소개서 내용:
            {cover_letter}
            """

    prompt = f"""
        너는 면접 질문 생성기다.

        규칙:
        - 질문 개수: {count}개
        - 질문만 생성 (설명 금지)
        - 반드시 순수 JSON 배열만 출력
        - ```json 같은 코드블럭 절대 금지
        - 다른 어떤 텍스트도 출력 금지
        - 각 질문은 1문장
        - 면접 흐름에 맞게 다양하게 구성
        
        출력 형식 예시:
        ["질문1", "질문2", "질문3"]

        직무: {job}
        
        {cover_letter_prompt}

        질문 생성:
        """

    return prompt