#!/usr/bin/env python3
"""4-2.mp4만 Faster-Whisper STT로 재평가한다."""

from pathlib import Path
import csv
import sys
import tempfile

# 기존 evaluate_stt.py의 함수/설정을 그대로 사용
from evaluate_stt import (
    WHISPER_MODEL_SIZE,
    WHISPER_COMPUTE_TYPE,
    normalize_text,
    text_for_cer,
    levenshtein_counts,
    extract_audio,
    transcribe,
)


TARGET_FILE = "4-2.mp4"


def get_ground_truth(csv_path: Path) -> str:
    """CSV에서 4-2.mp4의 Ground Truth만 가져온다."""

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            if row["file"].strip() == TARGET_FILE:
                return row["ground_truth"]

    raise ValueError(
        f"{TARGET_FILE}의 Ground Truth를 찾을 수 없습니다."
    )


def main() -> int:

    evaluation_dir = Path(__file__).resolve().parent

    video_path = (
        evaluation_dir
        / "videos"
        / TARGET_FILE
    )

    csv_path = (
        evaluation_dir
        / "interview_stt_ground_truth.csv"
    )

    if not video_path.is_file():
        print(
            f"오류: 영상을 찾을 수 없습니다: {video_path}",
            file=sys.stderr,
        )
        return 1

    try:
        ground_truth = get_ground_truth(csv_path)

        from faster_whisper import WhisperModel

        print(
            "Faster-Whisper 모델 로드: "
            f"model={WHISPER_MODEL_SIZE}, "
            "device=cpu, "
            f"compute_type={WHISPER_COMPUTE_TYPE}"
        )

        model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device="cpu",
            compute_type=WHISPER_COMPUTE_TYPE,
        )

        print(f"\n재평가 시작: {TARGET_FILE}")

        with tempfile.TemporaryDirectory(
            prefix="interviewfit_stt_4_2_"
        ) as temp_dir:

            audio_path = (
                Path(temp_dir)
                / "4-2.wav"
            )

            # 영상에서 음성 추출
            extract_audio(
                video_path,
                audio_path,
            )

            # 기존과 동일한 설정으로 STT
            raw_stt_result = transcribe(
                model,
                audio_path,
            )

        normalized_ground_truth = normalize_text(
            ground_truth
        )

        normalized_stt_result = normalize_text(
            raw_stt_result
        )

        reference = text_for_cer(
            ground_truth
        )

        hypothesis = text_for_cer(
            raw_stt_result
        )

        counts = levenshtein_counts(
            reference,
            hypothesis,
        )

        cer = (
            counts.distance
            / len(reference)
        )

        print("\n===== 4-2 STT 재평가 결과 =====")

        print(
            f"Ground Truth: "
            f"{normalized_ground_truth}"
        )

        print(
            f"STT Result: "
            f"{normalized_stt_result}"
        )

        print(
            f"S: {counts.substitutions}"
        )

        print(
            f"D: {counts.deletions}"
        )

        print(
            f"I: {counts.insertions}"
        )

        print(
            f"Edit Distance: "
            f"{counts.distance}"
        )

        print(
            f"Ground Truth Length: "
            f"{len(reference)}"
        )

        print(
            f"CER: {cer:.2%}"
        )

        return 0

    except Exception as exc:

        print(
            f"오류: {exc}",
            file=sys.stderr,
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(main())