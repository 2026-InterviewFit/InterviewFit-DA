def create_followup_prompt(
    job: str,
    original_question: str,
    answer: str,
    interview_style: str = "normal",
) -> str:
    style_guide_map = {
        "normal": "전문적이고 객관적인 톤으로",
        "friendly": "부드럽고 격려하는 톤으로",
        "pressure": "날카롭고 검증하는 톤으로 (다만 무례하지 않게)",
    }
    style_guide = style_guide_map.get(interview_style, "전문적이고 객관적인 톤으로")

    prompt = f"""
        너는 면접 꼬리질문 생성기다.

        지원자가 면접 질문에 답변한 내용을 분석하여
        답변의 깊이를 검증할 수 있는 꼬리질문 1개를 생성한다.

        규칙:
        - 꼬리질문은 정확히 1개만 생성
        - 면접 스타일: {style_guide}
        - 반드시 순수 JSON 객체만 출력
        - ```json 같은 코드블럭 절대 금지
        - 다른 어떤 텍스트도 출력 금지
        - 꼬리질문은 1문장으로 작성

        꼬리질문 작성 지침:
        - 답변에서 모호하거나 추상적인 표현이 있다면 구체화를 요구
        - 답변에 경험이 언급되었다면 구체적인 상황·행동·결과를 검증
        - 답변에 기술이 언급되었다면 그 기술에 대한 깊이 있는 이해를 검증
        - 답변이 직무와의 관련성이 부족하면 직무 연관성을 묻는 질문
        - 단순히 같은 질문을 반복하지 말 것

        출력 형식 예시:
        {{"follow_up_question": "쿼리 최적화는 구체적으로 어떤 방식으로 진행하셨고, 성능 개선 수치는 어느 정도였나요?"}}

        직무: {job}
        원본 질문: {original_question}
        지원자 답변: {answer}

        꼬리질문 생성:
        """

    return prompt