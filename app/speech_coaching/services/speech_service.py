import os
import yt_dlp
import librosa
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from faster_whisper import WhisperModel

# 모델 로드
print("무료 Whisper AI 모델을 로드하는 중입니다. 잠시만 기다려주세요...")
model = WhisperModel("base", device="cpu", compute_type="float32")
print("AI 모델 로드 완료!")


def download_youtube_video(youtube_url: str, output_name: str = "interview") -> str:
    try:
        print(f"\n유튜브 영상 다운로드 시작: {youtube_url}")
        ydl_opts = {'format': 'best[ext=mp4]/best', 'outtmpl': f'{output_name}.%(ext)s', 'quiet': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        expected_filename = f"{output_name}.mp4"
        if os.path.exists(expected_filename):
            print(f"유튜브 영상 다운로드 성공: {expected_filename}")
            return expected_filename
        return expected_filename
    except Exception as e:
        print(f"유튜브 다운로드 실패: {str(e)}")
        return ""


def extract_audio_and_get_duration(video_path: str, output_audio_path: str):
    if not os.path.exists(video_path):
        print(f"에러: {video_path} 경로에 영상 파일이 없습니다!")
        return False, 0.0
    try:
        print(f"영상을 분석하는 중: {video_path}")
        video_clip = VideoFileClip(video_path)
        duration_seconds = video_clip.duration
        print(f"영상 총 길이 확인됨: {duration_seconds:.2f}초")
        print(f"오디오 스트림 인코딩 및 변환 중 -> {output_audio_path}")
        video_clip.audio.write_audiofile(output_audio_path, codec='pcm_s16le', logger=None)
        video_clip.close()
        return True, duration_seconds
    except Exception as e:
        print(f"오디오 추출 및 시간 측정 실패: {str(e)}")
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
    except Exception as e:
        print(f"리브로사 음성 신호 분석 실패: {str(e)}")
        return {"total_silence_seconds": 0.0, "critical_silence_count": 0}


def run_free_whisper_stt(audio_path: str) -> str:
    try:
        segments, info = model.transcribe(audio_path, beam_size=5, language="ko")
        return " ".join([s.text for s in segments]).strip()
    except Exception as e:
        return f"STT 변환 실패: {str(e)}"


def analyze_speech_metrics(stt_text: str, duration_seconds: float, silence_report: dict) -> dict:
    padded_text = f" {stt_text} "
    filler_targets = ["음", "어", "그", "이제", "약간", "저"]
    filler_counts = {word: padded_text.count(f" {word} ") + padded_text.count(f" {word},") for word in filler_targets}
    total_filler_count = sum(filler_counts.values())
    speech_rate_bps = len(stt_text.replace(" ", "")) / duration_seconds if duration_seconds > 0 else 0.0

    if speech_rate_bps < 2.5:
        pace_eval, pace_feedback = "너무 느림", "답변 속도가 느려 지루한 인상을 줄 수 있습니다."
    elif 2.5 <= speech_rate_bps <= 4.2:
        pace_eval, pace_feedback = "적절함", "가장 안정적이고 신뢰감을 주는 속도입니다."
    else:
        pace_eval, pace_feedback = "너무 빠름", "답변 속도가 너무 빨라 면접관이 내용을 놓칠 우려가 있습니다."

    return {
        "text_script": stt_text,
        "filler_word_analysis": {"details": filler_counts, "total_count": total_filler_count},
        "speech_pace_analysis": {"evaluation": pace_eval, "feedback": pace_feedback,
                                 "characters_per_second": round(speech_rate_bps, 2)},
        "silence_analysis": silence_report
    }


def process_interview_analysis(youtube_url: str):
    temp_video = "interview.mp4"
    temp_audio = "extracted_voice.wav"

    path = download_youtube_video(youtube_url, "interview")
    if not path: return None

    audio_success, duration = extract_audio_and_get_duration(path, temp_audio)
    if not audio_success: return None

    print("\nLibrosa 음성 파형 정적 탐지 가동...")
    silence = analyze_silence_with_librosa(temp_audio)
    print("\n인공지능 자막 대본 추출 엔진 가동...")
    stt = run_free_whisper_stt(temp_audio)
    print("\n데이터 명세 연산 시작...")
    report = analyze_speech_metrics(stt, duration, silence)

    if os.path.exists(temp_audio): os.remove(temp_audio); print("소리파일제거")
    if os.path.exists(path): os.remove(path); print("시스템 하드디스크 임시 휘발성 파일 자동 청소 완료!")

    return report