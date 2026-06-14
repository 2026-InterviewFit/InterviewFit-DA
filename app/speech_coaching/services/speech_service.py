import os
import yt_dlp
import logging
import librosa
from moviepy.video.io.VideoFileClip import VideoFileClip
from faster_whisper import WhisperModel
from . import ai_evaluator_service
from ..infra.spring_client import send_to_spring_server
from ..infra import s3_client

# 로깅 및 모델 설정
logger = logging.getLogger(__name__)

print("무료 Whisper AI 모델을 로드하는 중입니다. 잠시만 기다려주세요...")
model = WhisperModel("base", device="cpu", compute_type="float32")
print("AI 모델 로드 완료!")


def download_youtube_video(youtube_url: str, output_name: str) -> str:
    try:
        ydl_opts = {'format': 'best[ext=mp4]/best', 'outtmpl': f'{output_name}.%(ext)s', 'quiet': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        return f"{output_name}.mp4"
    except Exception as e:
        logger.error(f"유튜브 다운로드 실패: {str(e)}")
        return ""


def extract_audio_and_get_duration(video_path: str, output_audio_path: str):
    try:
        video_clip = VideoFileClip(video_path)
        duration_seconds = video_clip.duration
        video_clip.audio.write_audiofile(output_audio_path, codec='pcm_s16le', logger=None)
        video_clip.close()
        return True, duration_seconds
    except Exception as e:
        logger.error(f"오디오 추출 실패: {e}")
        return False, 0.0


def analyze_silence_with_librosa(audio_path: str) -> dict:
    try:
        y, sr = librosa.load(audio_path, sr=None)
        non_silent_intervals = librosa.effects.split(y, top_db=30)
        total_duration = len(y) / sr
        active_duration = sum([end - start for start, end in non_silent_intervals]) / sr
        total_silence_seconds = max(0.0, total_duration - active_duration)
        critical_silence_count = 0
        last_end_sec = 0.0
        for start, end in non_silent_intervals:
            if (start / sr) - last_end_sec >= 3.0: critical_silence_count += 1
            last_end_sec = end / sr
        if total_duration - last_end_sec >= 3.0: critical_silence_count += 1
        return {"total_silence_seconds": round(total_silence_seconds, 2),
                "critical_silence_count": critical_silence_count}
    except Exception:
        return {"total_silence_seconds": 0.0, "critical_silence_count": 0}


def run_free_whisper_stt(audio_path: str) -> str:
    try:
        segments, _ = model.transcribe(audio_path, beam_size=5, language="ko")
        return " ".join([s.text for s in segments]).strip()
    except Exception as e:
        logger.error(f"STT 변환 실패: {e}")
        return "STT 변환 실패"


def analyze_speech_metrics(stt_text: str, duration_seconds: float, silence_report: dict) -> dict:
    padded_text = f" {stt_text} "
    filler_targets = ["음", "어", "그", "이제", "약간", "저"]
    filler_counts = {word: padded_text.count(f" {word} ") + padded_text.count(f" {word},") for word in filler_targets}
    total_filler_count = sum(filler_counts.values())
    speech_rate_bps = len(stt_text.replace(" ", "")) / duration_seconds if duration_seconds > 0 else 0.0

    return {
        "text_script": stt_text,
        "filler_word_analysis": {"details": filler_counts, "total_count": total_filler_count},
        "speech_pace_analysis": {"characters_per_second": round(speech_rate_bps, 2)},
        "silence_analysis": silence_report
    }


def process_interview_analysis(source_type: str, source_value: str, job_category: str, interview_id: int):
    suffix = os.urandom(4).hex()
    temp_video = f"temp_video_{suffix}.mp4"
    temp_audio = f"temp_audio_{suffix}.wav"
    path = ""

    try:
        # 1. 소스 준비 (유튜브, MinIO, 로컬 파일)
        if source_type == 'youtube':
            path = download_youtube_video(source_value, f"interview_{suffix}")
            if not path: return None
        elif source_type == 'minio':
            bucket_name = os.getenv("MINIO_BUCKET")
            # 경로에서 버킷 이름이 포함되어 있으면 제거
            object_key = source_value.replace(f"{bucket_name}/", "", 1)

            logger.info(f"MinIO 다운로드 시도: bucket={bucket_name}, key={object_key}")
            s3_client.download_video_from_minio(object_key, temp_video)

            # 파일 존재 여부 및 용량 확인 (에러 방지 핵심!)
            if not os.path.exists(temp_video) or os.path.getsize(temp_video) == 0:
                return {"error": "MinIO에서 파일을 찾을 수 없거나 다운로드에 실패했습니다."}
            path = temp_video
        elif source_type == 'local':
            path = source_value
            if not os.path.exists(path): return {"error": "로컬 파일을 찾을 수 없습니다."}
        else:
            return {"error": "지원하지 않는 소스 타입입니다."}

        # 2. 분석 수행
        audio_success, duration = extract_audio_and_get_duration(path, temp_audio)
        if not audio_success: return {"error": "오디오 추출 실패"}

        silence = analyze_silence_with_librosa(temp_audio)
        stt = run_free_whisper_stt(temp_audio)
        report = analyze_speech_metrics(stt, duration, silence)

        # 3. Gemini 평가 추가
        report['ai_evaluation'] = ai_evaluator_service.evaluate_interview_content(stt, job_category)
        report['job_category'] = job_category
        report['interview_id'] = interview_id

        # 4. 스프링 전송
        send_to_spring_server(report)
        return report

    except Exception as e:
        logger.exception(f"분석 파이프라인 에러: {e}")
        return {"error": str(e)}
    finally:
        # 5. 파일 정리 (로컬 원본 제외)
        if os.path.exists(temp_audio): os.remove(temp_audio)
        if source_type != 'local' and os.path.exists(path): os.remove(path)