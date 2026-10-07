#!/usr/bin/env python3
"""실제 면접 영상의 Faster-Whisper STT 문자 오류율(CER)을 평가한다."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# app/speech_coaching/services/speech_service.py의 운영 설정과 동일하다.
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

STT_FAILURE_RESULT = "STT 변환 실패"
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov"}


@dataclass(frozen=True)
class EditCounts:
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0

    @property
    def distance(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    def __add__(self, other: "EditCounts") -> "EditCounts":
        return EditCounts(
            self.substitutions + other.substitutions,
            self.deletions + other.deletions,
            self.insertions + other.insertions,
        )


def normalize_text(text: str) -> str:
    """GT와 STT에 공통 적용할 표시용 정규화."""
    text = re.sub(r"\s+", " ", text.strip())

    text = "".join(
        char
        for char in text
        if not unicodedata.category(char).startswith("P")
    )

    return re.sub(r"\s+", " ", text).strip().casefold()


def text_for_cer(text: str) -> str:
    """공통 정규화 후 CER 계산에서 모든 공백을 제외한다."""
    return "".join(normalize_text(text).split())


def levenshtein_counts(
    reference: str,
    hypothesis: str,
) -> EditCounts:
    """Levenshtein DP로 최소 편집거리와 S/D/I의 한 정렬을 계산한다."""

    previous = [
        EditCounts(insertions=index)
        for index in range(len(hypothesis) + 1)
    ]

    for ref_index, ref_char in enumerate(reference, start=1):

        current = [
            EditCounts(deletions=ref_index)
        ]

        for hyp_index, hyp_char in enumerate(hypothesis, start=1):

            if ref_char == hyp_char:
                current.append(previous[hyp_index - 1])
                continue

            substitution = (
                previous[hyp_index - 1]
                + EditCounts(substitutions=1)
            )

            deletion = (
                previous[hyp_index]
                + EditCounts(deletions=1)
            )

            insertion = (
                current[hyp_index - 1]
                + EditCounts(insertions=1)
            )

            # 최소거리가 같은 경우 S, D, I 순으로 선택
            current.append(
                min(
                    (substitution, deletion, insertion),
                    key=lambda value: value.distance,
                )
            )

        previous = current

    return previous[-1]


def extract_audio(
    video_path: Path,
    output_audio_path: Path,
) -> None:
    """서비스와 동일하게 16 kHz mono PCM WAV를 생성한다."""

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_audio_path),
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )

    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg를 찾을 수 없습니다. ffmpeg를 먼저 설치해주세요."
        ) from exc

    if result.returncode != 0:

        detail = result.stderr.strip().splitlines()

        message = (
            detail[-1]
            if detail
            else "알 수 없는 ffmpeg 오류"
        )

        raise RuntimeError(
            f"오디오 추출 실패: {message}"
        )


def join_whisper_segments(
    segments: Iterable[object],
) -> str:
    """Whisper segment 결과를 하나의 문자열로 합친다."""

    texts = []

    for segment in segments:

        text = (
            getattr(segment, "text", "") or ""
        ).strip()

        if text:
            texts.append(text)

    return " ".join(texts).strip()


def transcribe(
    model: object,
    audio_path: Path,
) -> str:
    """서비스 run_free_whisper_stt와 동일한 1차/재시도 설정을 사용한다."""

    # 1차 STT
    segments, _ = model.transcribe(
        str(audio_path),
        beam_size=5,
        language="ko",
        condition_on_previous_text=False,
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    )

    text = join_whisper_segments(segments)

    if text:
        return text

    # STT 결과가 비었을 경우 재시도
    segments, _ = model.transcribe(
        str(audio_path),
        beam_size=5,
        language=None,
        condition_on_previous_text=False,
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        no_speech_threshold=0.8,
        log_prob_threshold=-2.0,
    )

    return join_whisper_segments(segments)


def read_ground_truth(
    csv_path: Path,
) -> list[tuple[str, str]]:
    """
    interview_stt_ground_truth.csv 읽기

    필요한 컬럼:
    file, ground_truth
    """

    try:

        with csv_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:

            reader = csv.DictReader(file)

            if (
                reader.fieldnames is None
                or not {"file", "ground_truth"}.issubset(
                    reader.fieldnames
                )
            ):
                raise ValueError(
                    "interview_stt_ground_truth.csv에는 "
                    "file, ground_truth 컬럼이 필요합니다."
                )

            rows = [
                (
                    row["file"].strip(),
                    row["ground_truth"],
                )
                for row in reader
            ]

    except FileNotFoundError as exc:

        raise ValueError(
            f"Ground Truth 파일을 찾을 수 없습니다: {csv_path}"
        ) from exc

    if not rows:
        raise ValueError(
            "interview_stt_ground_truth.csv에 "
            "평가 데이터가 없습니다."
        )

    filenames = [
        filename
        for filename, _ in rows
    ]

    if any(
        not filename
        for filename in filenames
    ):
        raise ValueError(
            "file이 비어 있는 행이 있습니다."
        )

    if len(filenames) != len(set(filenames)):
        raise ValueError(
            "interview_stt_ground_truth.csv에 "
            "중복된 file이 있습니다."
        )

    for filename, ground_truth in rows:

        if not text_for_cer(ground_truth):
            raise ValueError(
                "정규화 후 Ground Truth가 "
                f"비어 있습니다: {filename}"
            )

    return rows


def resolve_video(
    videos_dir: Path,
    filename: str,
) -> Path:
    """CSV의 파일명과 실제 영상 파일을 연결한다."""

    videos_root = videos_dir.resolve()

    video_path = (
        videos_root / filename
    ).resolve()

    if video_path.parent != videos_root:
        raise ValueError(
            "file은 videos 폴더 바로 아래 "
            f"파일이어야 합니다: {filename}"
        )

    if (
        video_path.suffix.lower()
        not in SUPPORTED_VIDEO_EXTENSIONS
    ):
        raise ValueError(
            "지원하지 않는 영상 형식입니다"
            f"(.mp4, .mov만 지원): {filename}"
        )

    if not video_path.is_file():
        raise FileNotFoundError(
            f"평가 영상을 찾을 수 없습니다: {video_path}"
        )

    return video_path


def write_results(
    output_path: Path,
    results: list[dict[str, object]],
    total: EditCounts,
    total_length: int,
) -> None:
    """각 영상의 STT 평가 결과를 CSV로 저장한다."""

    fieldnames = [
        "filename",
        "ground_truth",
        "stt_result",
        "ground_truth_length",
        "substitutions",
        "deletions",
        "insertions",
        "edit_distance",
        "cer",
    ]

    overall_cer = (
        total.distance / total_length
    )

    summary = {
        "filename": "__OVERALL__",
        "ground_truth": "",
        "stt_result": "",
        "ground_truth_length": total_length,
        "substitutions": total.substitutions,
        "deletions": total.deletions,
        "insertions": total.insertions,
        "edit_distance": total.distance,
        "cer": f"{overall_cer:.6f}",
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)
        writer.writerow(summary)


def parse_args() -> argparse.Namespace:
    """평가에 사용할 파일 경로 설정."""

    evaluation_dir = (
        Path(__file__).resolve().parent
    )

    parser = argparse.ArgumentParser(
        description="Faster-Whisper STT CER 평가"
    )

    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=evaluation_dir / "videos",
    )

    # ★ 기존 ground_truth.csv → 변경
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=(
            evaluation_dir
            / "interview_stt_ground_truth.csv"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            evaluation_dir
            / "stt_evaluation_results.csv"
        ),
    )

    return parser.parse_args()


def main() -> int:

    args = parse_args()

    try:

        # Ground Truth 읽기
        evaluation_rows = read_ground_truth(
            args.ground_truth
        )

        # 영상 존재 여부를 Whisper 로딩 전에 확인
        resolved_rows = [
            (
                filename,
                ground_truth,
                resolve_video(
                    args.videos_dir,
                    filename,
                ),
            )
            for filename, ground_truth
            in evaluation_rows
        ]

        try:
            from faster_whisper import WhisperModel

        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper가 설치되어 있지 않습니다."
            ) from exc

        print(
            f"Faster-Whisper 모델 로드: "
            f"model={WHISPER_MODEL_SIZE}, "
            f"device=cpu, "
            f"compute_type={WHISPER_COMPUTE_TYPE}"
        )

        model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device="cpu",
            compute_type=WHISPER_COMPUTE_TYPE,
        )

        results: list[
            dict[str, object]
        ] = []

        total = EditCounts()
        total_length = 0

        with tempfile.TemporaryDirectory(
            prefix="interviewfit_stt_evaluation_"
        ) as temp_dir:

            total_videos = len(resolved_rows)

            for index, (
                filename,
                ground_truth,
                video_path,
            ) in enumerate(
                resolved_rows,
                start=1,
            ):

                print(
                    f"\n[{index}/{total_videos}] "
                    f"평가 시작: {filename}"
                )

                audio_path = (
                    Path(temp_dir)
                    / f"audio_{index:04d}.wav"
                )

                # 영상 → WAV
                extract_audio(
                    video_path,
                    audio_path,
                )

                # Faster-Whisper STT
                raw_stt_result = transcribe(
                    model,
                    audio_path,
                )

                if not raw_stt_result:

                    print(
                        f"경고: {filename}의 "
                        "STT 결과가 비어 있습니다.",
                        file=sys.stderr,
                    )

                # 정규화
                normalized_ground_truth = (
                    normalize_text(
                        ground_truth
                    )
                )

                normalized_stt_result = (
                    normalize_text(
                        raw_stt_result
                    )
                )

                # CER 계산용 문자열
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

                # 개별 결과 출력
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
                    f"CER: {cer:.2%}"
                )

                # 결과 저장
                results.append(
                    {
                        "filename": filename,
                        "ground_truth":
                            normalized_ground_truth,
                        "stt_result":
                            normalized_stt_result,
                        "ground_truth_length":
                            len(reference),
                        "substitutions":
                            counts.substitutions,
                        "deletions":
                            counts.deletions,
                        "insertions":
                            counts.insertions,
                        "edit_distance":
                            counts.distance,
                        "cer":
                            f"{cer:.6f}",
                    }
                )

                total += counts
                total_length += len(reference)

        # 전체 CER
        overall_cer = (
            total.distance
            / total_length
        )

        # CSV 결과 저장
        write_results(
            args.output,
            results,
            total,
            total_length,
        )

        print(
            "\n"
            "===== STT Evaluation Result ====="
        )

        print(
            f"평가 영상 수: "
            f"{len(results)}"
        )

        print(
            f"전체 Ground Truth 문자 수: "
            f"{total_length}"
        )

        print(
            f"Substitutions: "
            f"{total.substitutions}"
        )

        print(
            f"Deletions: "
            f"{total.deletions}"
        )

        print(
            f"Insertions: "
            f"{total.insertions}"
        )

        print(
            f"Overall CER: "
            f"{overall_cer:.2%}"
        )

        print(
            f"결과 CSV: "
            f"{args.output.resolve()}"
        )

        return 0

    except (
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:

        print(
            f"오류: {exc}",
            file=sys.stderr,
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(main())
