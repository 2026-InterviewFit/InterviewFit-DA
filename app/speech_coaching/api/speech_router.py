from fastapi import APIRouter
from app.speech_coaching.services import speech_service

router = APIRouter()


# 요청 예시: /speech/analyze?youtube_url=...&job_category=개발자
@router.post("/analyze")
async def analyze_interview(youtube_url: str, job_category: str = "개발자"):
    # 프론트에서 받은 job_category를 서비스 함수로 전달
    report = speech_service.process_interview_analysis(youtube_url, job_category)

    if not report:
        return {"error": "분석 중 오류 발생"}

    # 터미널 출력용 (디버깅)
    print("\n==================================================")
    print(f"[최종 조현진 담당 AI 멀티모달 스피치 코칭 리포트 - 직무: {job_category}]")
    print("==================================================")
    print(f"1. 추출 대본 스크립트:\n{report['text_script']}\n")
    print(f"2. [AI 심층 평가 - STAR 구조 및 키워드]:\n{report.get('ai_evaluation', '분석 실패')}\n")
    print("3. 무의미한 습관어(간투어) 빈도 통계:")
    print(f"   • 총 사용 횟수: {report['filler_word_analysis']['total_count']}회")
    print(f"   • 세부 내역: {report['filler_word_analysis']['details']}\n")
    print("4. 말하기 속도(초당 발화량)의 적절성:")
    print(f"   • AI 종합 판정 수치: 【 {report['speech_pace_analysis']['evaluation']} 】")
    print(f"   • 코칭 시스템 최종 피드백: {report['speech_pace_analysis']['feedback']}\n")
    print("5. 리브로사 기반 긴장도(침묵 구간) 분석:")
    print(f"   • [핵심 지표] 3초 이상 치명적 정적 발생 횟수: {report['silence_analysis']['critical_silence_count']}회")
    print("==================================================")

    return report