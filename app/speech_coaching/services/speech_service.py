import os
import logging
import subprocess
import re
from collections import Counter
import librosa
import yt_dlp
from faster_whisper import WhisperModel
from typing import Optional
from app.speech_coaching.services import ai_evaluator_service
from app.speech_coaching.infra import s3_client
from app.speech_coaching.schemas.speech_schema import (
    SpeechReportResponse, AiEvaluation,
    FillerWordAnalysis, RepeatedExpression, SpeechPaceAnalysis,
    SilenceAnalysis,
)
from app.common.exceptions.custom_exception import CustomException
from app.common.exceptions.error_code import ErrorCode
from app.common.exceptions.status_code import StatusCode

logger = logging.getLogger(__name__)

# Whisper AI 모델 로드 (서버 시작 시 1회)
logger.info("무료 Whisper AI 모델을 로드하는 중입니다...")
model = WhisperModel("base", device="cpu", compute_type="float32")
logger.info("AI 모델 로드 완료!")


# 유튜브 영상 다운로드
def download_youtube_video(youtube_url: str, output_name: str) -> str:
    try:
        ydl_opts = {
            "format": "best[ext=mp4]/best",
            "outtmpl": f"{output_name}.%(ext)s",
            "quiet": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        return f"{output_name}.mp4"
    except Exception as e:
        logger.error("유튜브 다운로드 실패: %s", e)
        return ""


# 영상에서 오디오 추출 + 길이 계산
def extract_audio_and_get_duration(video_path: str, output_audio_path: str):
    try:
        command = [
            "ffmpeg", "-y", "-i", video_path, "-vn",
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            output_audio_path,
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            logger.error("ffmpeg 오디오 추출 실패: %s", result.stderr)
            return False, 0.0

        y, sr = librosa.load(output_audio_path, sr=None)
        duration_seconds = len(y) / sr if sr else 0.0
        return True, duration_seconds
    except Exception as e:
        logger.error("오디오 추출 실패: %s", e)
        return False, 0.0


# 침묵 구간 및 침묵 비율 분석 (음량 분석 제외됨)
def analyze_audio_metrics(audio_path: str) -> dict:
    try:
        y, sr = librosa.load(audio_path, sr=None)
        total_duration = len(y) / sr if sr else 0.0

        # 침묵 분석
        non_silent_intervals = librosa.effects.split(y, top_db=30)
        active_duration = sum([end - start for start, end in non_silent_intervals]) / sr if sr else 0.0
        total_silence_seconds = max(0.0, total_duration - active_duration)
        pause_rate_percent = (total_silence_seconds / total_duration * 100) if total_duration > 0 else 0.0

        # 3초 이상 끊긴 구간(치명적 침묵) 카운트
        critical_silence_count = 0
        last_end_sec = 0.0
        for start, end in non_silent_intervals:
            if (start / sr) - last_end_sec >= 3.0:
                critical_silence_count += 1
            last_end_sec = end / sr
        if total_duration - last_end_sec >= 3.0:
            critical_silence_count += 1

        return {
            "total_silence_seconds": round(total_silence_seconds, 2),
            "pause_rate_percent": round(pause_rate_percent, 2),
            "critical_silence_count": critical_silence_count,
        }
    except Exception as e:
        logger.error("침묵 분석 실패: %s", e)
        return {
            "total_silence_seconds": 0.0,
            "pause_rate_percent": 0.0,
            "critical_silence_count": 0,
        }


# 연속으로 같은 단어가 반복되는 패턴 탐지
def find_consecutive_repeats(text: str, min_repeat: int = 2) -> list[dict]:
    pattern = re.compile(r"([가-힣A-Za-z]+)(?:[,]?\s+\1\b)+")
    results = []
    for m in pattern.finditer(text):
        word = m.group(1)
        count = len(re.findall(re.escape(word), m.group(0)))
        if count >= min_repeat:
            results.append({"expression": word, "count": count})
    return results


# 반복 표현 탐지
def detect_repeated_expressions(stt_text: str, min_length: int = 2, min_count: int = 2, top_n: int = 5) -> list[dict]:
    try:
        stopwords = {
            "그리고", "그래서", "그러나", "하지만", "그런데", "또한", "근데",
            "제가", "저는", "나는", "우리", "이것", "저것", "그것",
            "이런", "저런", "그런", "것", "수", "등", "때", "좀", "막", "같습니다",
        }
        filler_targets = {"음", "어", "그", "이제", "약간", "저"}

        consecutive = find_consecutive_repeats(stt_text, min_repeat=min_count)
        consecutive_words = {item["expression"] for item in consecutive}

        words = re.findall(r"[가-힣A-Za-z]+", stt_text)
        candidates = [
            w for w in words
            if w not in filler_targets
            and w not in stopwords
            and w not in consecutive_words
            and len(w) >= min_length
        ]
        counter = Counter(candidates)
        frequent = [
            {"expression": word, "count": count}
            for word, count in counter.most_common()
            if count >= min_count
        ]

        return (consecutive + frequent)[:top_n]
    except Exception as e:
        logger.error("반복 표현 탐지 실패: %s", e)
        return []


# Whisper STT 변환
def _join_whisper_segments(segments) -> str:
    """Whisper segment iterator를 안전하게 하나의 문자열로 합친다."""
    texts = []

    for segment in segments:
        text = (segment.text or "").strip()
        if text:
            texts.append(text)

    return " ".join(texts).strip()


def run_free_whisper_stt(audio_path: str) -> str:
    """
    Whisper STT 변환.

    1차는 한국어 고정으로 변환하고,
    결과가 비어 있으면 2차에서 언어 자동 감지 + 완화된 no-speech 기준으로
    한 번 더 시도한다.

    실제 무음 파일은 최종적으로 빈 문자열을 반환하며,
    라우터가 이를 422로 변환한다.
    """
    try:
        segments, _ = model.transcribe(
            audio_path,
            beam_size=5,
            language="ko",
            condition_on_previous_text=False,
            temperature=0.0,
        )

        text = _join_whisper_segments(segments)

        if text:
            return text

        logger.warning(
            "1차 Whisper STT 결과가 비어 있어 재시도합니다: %s",
            audio_path,
        )

        segments, _ = model.transcribe(
            audio_path,
            beam_size=5,
            language=None,
            condition_on_previous_text=False,
            temperature=0.0,
            no_speech_threshold=0.8,
            log_prob_threshold=-2.0,
        )

        return _join_whisper_segments(segments)

    except Exception as e:
        logger.error("STT 변환 실패: %s", e)
        return "STT 변환 실패"


# 스피치 지표 종합 계산
def analyze_speech_metrics(stt_text: str, duration_seconds: float) -> dict:
    padded_text = f" {stt_text} "
    filler_targets = ["음", "어", "그", "이제", "약간", "저"]
    filler_counts = {word: padded_text.count(f" {word} ") + padded_text.count(f" {word},") for word in filler_targets}
    total_filler_count = sum(filler_counts.values())

    cps = len(stt_text.replace(" ", "")) / duration_seconds if duration_seconds > 0 else 0.0
    pace_eval = "적절함"
    if cps < 2.0:
        pace_eval = "너무 느림"
    elif cps > 6.0:
        pace_eval = "너무 빠름"

    return {
        "text_script": stt_text,
        "total_filler_count": total_filler_count,
        "filler_word_analysis": {"details": filler_counts, "total_count": total_filler_count},
        "repeated_expressions": detect_repeated_expressions(stt_text),
        "speech_pace_analysis": {
            "evaluation": pace_eval,
            "feedback": f"초당 {round(cps, 2)}자 속도입니다.",
            "characters_per_second": round(cps, 2),
        },
    }


# 전체 분석 파이프라인 (답변별 식별자 포함, 음량 분석 제외)
def process_interview_analysis(
    source_type: str,
    source_value: str,
    job_category: str,
    interview_id: int,
    interview_answer_id: Optional[int] = None,
    question_id: Optional[int] = None,
    original_question: Optional[str] = None
):
    suffix = os.urandom(4).hex()
    temp_video = f"temp_video_{suffix}.mp4"
    temp_audio = f"temp_audio_{suffix}.wav"
    video_path = ""

    try:
        # 1. 소스 타입별 영상 확보
        if source_type == "youtube":
            video_path = download_youtube_video(source_value, f"interview_{suffix}")
        elif source_type == "minio":
            video_path = s3_client.download_video_from_minio(source_value, temp_video)
        elif source_type == "local":
            video_path = source_value
        else:
            raise CustomException(
                status=StatusCode.BAD_REQUEST,
                code=ErrorCode.INVALID_SOURCE_TYPE,
                message=f"지원하지 않는 source_type입니다: {source_type}"
            )

        if not video_path or not os.path.exists(video_path):
            raise CustomException(
                status=StatusCode.BAD_REQUEST,
                code=ErrorCode.VIDEO_DOWNLOAD_FAILED,
                message="영상을 확보하지 못했습니다. source_value를 확인해주세요."
            )

        # 2. 분석 파이프라인
        extracted, duration = extract_audio_and_get_duration(video_path, temp_audio)
        if not extracted:
            raise CustomException(
                status=StatusCode.INTERNAL_SERVER_ERROR,
                code=ErrorCode.AUDIO_EXTRACTION_FAILED,
                message="오디오 추출에 실패했습니다."
            )

        audio_metrics = analyze_audio_metrics(temp_audio)
        stt_text = run_free_whisper_stt(temp_audio)
        metrics = analyze_speech_metrics(stt_text, duration)

        # 3. Gemini 평가
        ai_res = ai_evaluator_service.evaluate_interview_content(
            stt_text,
            job_category,
            original_question,
        )
        if not isinstance(ai_res, dict):
            raise CustomException(
                status=StatusCode.INTERNAL_SERVER_ERROR,
                code=ErrorCode.AI_EVALUATION_FAILED,
                message="Gemini 평가 결과 형식이 올바르지 않습니다."
            )
        try:
            ai_evaluation = AiEvaluation(**ai_res)
        except (TypeError, ValueError) as e:
            logger.warning(
                "Gemini 평가 결과 매핑 실패: error_type=%s",
                type(e).__name__,
            )
            raise CustomException(
                status=StatusCode.INTERNAL_SERVER_ERROR,
                code=ErrorCode.AI_EVALUATION_FAILED,
                message="Gemini 평가 결과 형식이 올바르지 않습니다."
            ) from e

        # 4. 음량 분석을 제외한 응답 스키마 매핑 및 반환
        return SpeechReportResponse(
            interview_id=interview_id,
            interview_answer_id=interview_answer_id,
            question_id=question_id,
            original_question=original_question,
            job_category=job_category,
            text_script=metrics["text_script"],
            filler_word_analysis=FillerWordAnalysis(**metrics["filler_word_analysis"]),
            repeated_expressions=[RepeatedExpression(**r) for r in metrics["repeated_expressions"]],
            speech_pace_analysis=SpeechPaceAnalysis(**metrics["speech_pace_analysis"]),
            silence_analysis=SilenceAnalysis(
                total_silence_seconds=audio_metrics["total_silence_seconds"],
                pause_rate_percent=audio_metrics["pause_rate_percent"],
                critical_silence_count=audio_metrics["critical_silence_count"],
            ),
            ai_evaluation=ai_evaluation,
            #model_answer=ai_res.get("model_answer", "")
        )
    except CustomException:
        raise
    except Exception as e:
        logger.exception("분석 파이프라인 전체 에러")
        raise CustomException(
            status=StatusCode.INTERNAL_SERVER_ERROR,
            code=ErrorCode.SPEECH_ANALYSIS_FAILED,
            message=str(e)
        )
    finally:
        if os.path.exists(temp_audio): os.remove(temp_audio)
        if source_type != "local" and video_path and os.path.exists(video_path): os.remove(video_path)