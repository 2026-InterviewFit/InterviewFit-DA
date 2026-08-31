from typing import List, Tuple
import platform

import cv2
import mediapipe as mp
import numpy as np


# ============================
# MediaPipe Face Mesh 랜드마크 인덱스
# ============================

LEFT_EYE_INDEXES = [
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246,
]

RIGHT_EYE_INDEXES = [
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398,
]

# EAR 계산용 6포인트
# 순서: [P1(외측), P2(상단1), P3(상단2), P4(내측), P5(하단2), P6(하단1)]
LEFT_EYE_EAR_INDEXES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR_INDEXES = [362, 385, 387, 263, 373, 380]


# ============================
# 깜빡임 감지 기준
# ============================

EAR_THRESHOLD = 0.24
EAR_CONSEC_FRAMES = 2

# 일반적인 30fps 면접 영상은 모든 프레임을 그대로 분석한다.
# 60/120fps처럼 고FPS 영상만 최대 약 30fps 수준으로 줄여 CPU 폭주를 막는다.
MAX_PROCESS_FPS = 30.0

# FaceMesh에는 원본 고해상도 프레임 대신 가로 최대 640px 이미지를 전달한다.
# EAR은 눈의 거리 비율이므로 종횡비를 유지해 축소하면 판정 원리는 동일하다.
MAX_PROCESS_WIDTH = 640



def _calculate_ear(eye_points: List[Tuple[float, float]]) -> float:
    """
    Eye Aspect Ratio 계산

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """

    if len(eye_points) != 6:
        raise ValueError("EAR 계산에는 6개의 눈 좌표가 필요합니다.")

    p1, p2, p3, p4, p5, p6 = [
        np.array(point, dtype=np.float32)
        for point in eye_points
    ]

    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)

    if horizontal == 0:
        return 0.0

    return float((vertical_1 + vertical_2) / (2.0 * horizontal))


def _landmark_to_pixel_point(landmark, width: int, height: int) -> Tuple[float, float]:
    """MediaPipe 정규화 좌표를 실제 pixel 좌표로 변환한다."""

    return landmark.x * width, landmark.y * height


def _resize_for_processing(frame):
    """
    FaceMesh 입력만 축소한다.

    원본의 종횡비를 그대로 유지하므로 EAR 계산 비율에는 영향을 주지 않는다.
    """

    height, width = frame.shape[:2]

    if width <= MAX_PROCESS_WIDTH:
        return frame

    scale = MAX_PROCESS_WIDTH / float(width)
    resized_height = max(1, int(round(height * scale)))

    return cv2.resize(
        frame,
        (MAX_PROCESS_WIDTH, resized_height),
        interpolation=cv2.INTER_AREA,
    )


def _open_video_capture(video_path: str):
    """
    OpenCV 기본 backend로 먼저 영상을 열고,
    실패하면 FFmpeg backend를 한 번 더 시도한다.

    OS별 OpenCV backend 차이로 WebM이 열리지 않는 경우를 완화한다.
    """

    cap = cv2.VideoCapture(video_path)

    if cap.isOpened():
        return cap

    cap.release()

    if hasattr(cv2, "CAP_FFMPEG"):
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if cap.isOpened():
            return cap
        cap.release()

    raise ValueError(
        "비디오 파일을 열 수 없습니다. "
        f"path={video_path}, "
        f"opencv={getattr(cv2, '__version__', 'unknown')}, "
        f"os={platform.system()} {platform.release()}"
    )


def _normalize_fps(raw_fps: float | None) -> float:
    """WebM 메타데이터의 비정상 FPS를 안전한 값으로 보정한다."""

    if raw_fps is None or raw_fps <= 0 or raw_fps > 240:
        return 30.0

    return float(raw_fps)


def _processing_frame_step(fps: float) -> int:
    """30fps 이하는 유지하고 고FPS 영상만 샘플링한다."""

    if fps <= MAX_PROCESS_FPS:
        return 1

    return max(1, int(round(fps / MAX_PROCESS_FPS)))


def _print_video_diagnostics(
    video_path: str,
    cap,
    fps: float,
    metadata_total_frames: int,
    frame_step: int,
) -> None:
    """다른 PC에서 오류가 날 때 환경 차이를 바로 비교할 수 있도록 로그를 남긴다."""

    backend = "unknown"
    try:
        backend = cap.getBackendName()
    except Exception:
        pass

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    print("======================================")
    print("[EyeBlinking 영상 분석 환경]")
    print(f"path = {video_path}")
    print(f"os = {platform.system()} {platform.release()}")
    print(f"python = {platform.python_version()}")
    print(f"opencv = {getattr(cv2, '__version__', 'unknown')}")
    print(f"mediapipe = {getattr(mp, '__version__', 'unknown')}")
    print(f"videoBackend = {backend}")
    print(f"resolution = {width}x{height}")
    print(f"fps = {fps}")
    print(f"metadataTotalFrames = {metadata_total_frames}")
    print(f"frameStep = {frame_step}")
    print(f"maxProcessWidth = {MAX_PROCESS_WIDTH}")
    print(f"earThreshold = {EAR_THRESHOLD}")
    print(f"earConsecFrames = {EAR_CONSEC_FRAMES}")
    print("======================================")


def extract_eye_landmarks_from_image(image_bytes: bytes) -> dict:
    """이미지 bytes를 받아 왼쪽 눈 / 오른쪽 눈 랜드마크 좌표를 반환한다."""

    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("이미지를 읽을 수 없습니다.")

    image = _resize_for_processing(image)
    height, width, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        # Blink EAR에는 iris landmark가 필요하지 않아 False로 낮춘다.
        refine_landmarks=False,
        min_detection_confidence=0.5,
    ) as face_mesh:
        results = face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            raise ValueError("얼굴을 찾을 수 없습니다.")

        face_landmarks = results.multi_face_landmarks[0].landmark

        left_eye = [
            {
                "index": idx,
                "x": float(face_landmarks[idx].x),
                "y": float(face_landmarks[idx].y),
                "z": float(face_landmarks[idx].z),
                "pixel_x": int(face_landmarks[idx].x * width),
                "pixel_y": int(face_landmarks[idx].y * height),
            }
            for idx in LEFT_EYE_INDEXES
        ]

        right_eye = [
            {
                "index": idx,
                "x": float(face_landmarks[idx].x),
                "y": float(face_landmarks[idx].y),
                "z": float(face_landmarks[idx].z),
                "pixel_x": int(face_landmarks[idx].x * width),
                "pixel_y": int(face_landmarks[idx].y * height),
            }
            for idx in RIGHT_EYE_INDEXES
        ]

        left_eye_points = [
            _landmark_to_pixel_point(face_landmarks[i], width, height)
            for i in LEFT_EYE_EAR_INDEXES
        ]
        right_eye_points = [
            _landmark_to_pixel_point(face_landmarks[i], width, height)
            for i in RIGHT_EYE_EAR_INDEXES
        ]

        left_ear = _calculate_ear(left_eye_points)
        right_ear = _calculate_ear(right_eye_points)
        average_ear = (left_ear + right_ear) / 2.0

    return {
        "face_detected": True,
        "image_width": width,
        "image_height": height,
        "left_ear": round(left_ear, 4),
        "right_ear": round(right_ear, 4),
        "average_ear": round(average_ear, 4),
        "eye_status": "closed" if average_ear < EAR_THRESHOLD else "open",
        "ear_threshold": EAR_THRESHOLD,
        "left_eye_landmarks": left_eye,
        "right_eye_landmarks": right_eye,
    }


# 눈 깜빡임 빈도 평가는 긴장도를 직접 진단하는 기준이 아니라
# 면접 중 비언어적 응시 태도를 확인하기 위한 보조 지표로 사용한다.
#
# Doughty (2001)는 정상 성인의 대화(conversation) 상황에서
# 자발적 눈 깜빡임 빈도의 95% 신뢰구간을 약 10.5~32.5회/분으로 보고하였다.
# 본 프로젝트는 면접을 대화 중심 활동으로 보고 이 범위를 참고 기준으로 사용한다.
BLINK_REFERENCE_MIN_PER_MINUTE = 10.5
BLINK_REFERENCE_MAX_PER_MINUTE = 32.5


def _get_blink_evaluation(blinks_per_minute: float) -> str:
    """
    대화 상황에서 보고된 자발적 눈 깜빡임 빈도 범위를 기준으로
    비언어적 응시 태도에 대한 보조 피드백을 생성한다.

    이 평가는 긴장도나 의학적 상태를 직접 판정하지 않는다.
    """
    if blinks_per_minute < BLINK_REFERENCE_MIN_PER_MINUTE:
        return (
            "눈 깜빡임 빈도가 정상 성인의 일반적인 대화 상황에서 보고된 "
            "참고 범위(약 10.5~32.5회/분)보다 낮은 편입니다. "
            "장시간 시선을 고정하고 있지는 않은지 확인해보세요. "
            "개인차와 촬영 환경의 영향도 함께 고려해야 합니다."
        )

    if blinks_per_minute <= BLINK_REFERENCE_MAX_PER_MINUTE:
        return (
            "눈 깜빡임 빈도가 정상 성인의 대화 상황에서 보고된 "
            "일반적인 참고 범위(약 10.5~32.5회/분)에 해당합니다. "
            "비언어적 응시 태도 안정성을 확인하는 보조 지표로 활용할 수 있습니다."
        )

    return (
        "눈 깜빡임 빈도가 정상 성인의 일반적인 대화 상황에서 보고된 "
        "참고 범위(약 10.5~32.5회/분)보다 높은 편입니다. "
        "눈 깜빡임이 지나치게 빈번하지 않았는지 확인해보세요. "
        "개인차와 촬영 환경의 영향도 함께 고려해야 합니다."
    )


def _get_blink_score(blinks_per_minute: float) -> int:
    """
    눈 깜빡임 빈도를 사용자에게 직관적으로 보여주기 위한 프로젝트 내부 점수이다.

    참고 범위 자체는 Doughty (2001)의 대화 상황 연구값을 사용하지만,
    100점/80점이라는 점수는 논문에서 제시한 수치가 아니라
    서비스 UI를 위한 프로젝트 내부 정책이다.

    - 10.5~32.5회/분: 대화 상황 참고 범위 → 100점
    - 10.5회/분 미만 또는 32.5회/분 초과: 참고 범위 밖 → 80점
    """
    if BLINK_REFERENCE_MIN_PER_MINUTE <= blinks_per_minute <= BLINK_REFERENCE_MAX_PER_MINUTE:
        return 100

    return 80


def analyze_blinks_from_video(video_path: str) -> dict:
    """
    비디오 파일을 EAR 기반으로 분석한다.

    안정화 포인트:
    - 30fps 이하는 모든 프레임 분석
    - 고FPS만 최대 약 30fps로 샘플링
    - FaceMesh 입력 해상도는 가로 최대 640px
    - Blink에 불필요한 iris refinement 비활성화
    - OpenCV backend / FPS / 버전 진단 로그 출력
    """

    cap = _open_video_capture(video_path)

    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = _normalize_fps(raw_fps)
    metadata_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_step = _processing_frame_step(fps)

    _print_video_diagnostics(
        video_path,
        cap,
        fps,
        metadata_total_frames,
        frame_step,
    )

    mp_face_mesh = mp.solutions.face_mesh

    blink_count = 0
    consec_closed_frames = 0

    ear_history: List[float] = []
    blink_timestamps: List[float] = []
    closed_frame_indexes: List[int] = []

    # source frame 기준
    decoded_frame_count = 0

    # 실제 FaceMesh에 넣은 frame 기준
    processed_frame_count = 0
    face_detected_frames = 0

    try:
        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            # EAR landmark는 iris refinement가 없어도 제공된다.
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                source_frame_index = decoded_frame_count
                decoded_frame_count += 1

                if source_frame_index % frame_step != 0:
                    continue

                processed_frame_count += 1

                frame_for_process = _resize_for_processing(frame)
                height, width, _ = frame_for_process.shape

                rgb_frame = cv2.cvtColor(
                    frame_for_process,
                    cv2.COLOR_BGR2RGB,
                )
                results = face_mesh.process(rgb_frame)

                if results.multi_face_landmarks:
                    face_detected_frames += 1
                    landmarks = results.multi_face_landmarks[0].landmark

                    left_eye_points = [
                        _landmark_to_pixel_point(landmarks[i], width, height)
                        for i in LEFT_EYE_EAR_INDEXES
                    ]
                    right_eye_points = [
                        _landmark_to_pixel_point(landmarks[i], width, height)
                        for i in RIGHT_EYE_EAR_INDEXES
                    ]

                    left_ear = _calculate_ear(left_eye_points)
                    right_ear = _calculate_ear(right_eye_points)
                    avg_ear = (left_ear + right_ear) / 2.0

                    ear_history.append(avg_ear)

                    if avg_ear < EAR_THRESHOLD:
                        consec_closed_frames += 1
                        closed_frame_indexes.append(source_frame_index)
                    else:
                        if consec_closed_frames >= EAR_CONSEC_FRAMES:
                            blink_count += 1
                            timestamp = source_frame_index / fps
                            blink_timestamps.append(round(timestamp, 2))

                        consec_closed_frames = 0
                else:
                    # 얼굴을 놓친 순간을 눈이 다시 뜬 것으로 간주하면 false blink가 생길 수 있다.
                    # 따라서 얼굴 미검출은 blink 확정에 사용하지 않고 감김 연속 상태만 초기화한다.
                    consec_closed_frames = 0

            if consec_closed_frames >= EAR_CONSEC_FRAMES:
                blink_count += 1
                last_source_frame = max(0, decoded_frame_count - 1)
                timestamp = last_source_frame / fps if fps > 0 else 0.0
                blink_timestamps.append(round(timestamp, 2))

    finally:
        cap.release()

    if decoded_frame_count == 0:
        raise ValueError(
            "영상을 열었지만 디코딩된 프레임이 없습니다. "
            "WebM 코덱 또는 OpenCV/FFmpeg 설치 상태를 확인해주세요."
        )

    if processed_frame_count == 0:
        raise ValueError("분석할 수 있는 프레임이 없습니다.")

    # WebM은 CAP_PROP_FRAME_COUNT가 부정확한 환경이 있어 실제 디코딩 수로 길이를 계산한다.
    # 응답의 total_frames/fps는 실제 분석한 프레임 기준으로 맞춰서
    # Spring에서 face_detected_frames와 비율을 계산해도 일관되게 유지한다.
    effective_fps = fps / frame_step
    total_frames = processed_frame_count
    duration_seconds = decoded_frame_count / fps if fps > 0 else 0.0

    average_ear = float(np.mean(ear_history)) if ear_history else 0.0
    min_ear = float(np.min(ear_history)) if ear_history else 0.0
    max_ear = float(np.max(ear_history)) if ear_history else 0.0

    blinks_per_minute = (
        (blink_count / duration_seconds) * 60
        if duration_seconds > 0
        else 0.0
    )

    evaluation = _get_blink_evaluation(blinks_per_minute)
    score = _get_blink_score(blinks_per_minute)

    print("======================================")
    print("[EyeBlinking 분석 완료]")
    print(f"decodedFrames = {decoded_frame_count}")
    print(f"processedFrames = {processed_frame_count}")
    print(f"faceDetectedFrames = {face_detected_frames}")
    print(f"blinkCount = {blink_count}")
    print(f"blinksPerMinute = {round(blinks_per_minute, 2)}")
    print("======================================")

    return {
        "total_frames": total_frames,
        "fps": round(effective_fps, 2),
        "duration_seconds": round(duration_seconds, 2),
        "face_detected_frames": face_detected_frames,
        "blink_count": blink_count,
        "blinks_per_minute": round(blinks_per_minute, 2),
        "average_ear": round(average_ear, 4),
        "min_ear": round(min_ear, 4),
        "max_ear": round(max_ear, 4),
        "ear_threshold": EAR_THRESHOLD,
        "ear_consec_frames": EAR_CONSEC_FRAMES,
        "blink_timestamps": blink_timestamps,
        "closed_frame_count": len(closed_frame_indexes),
        "evaluation": evaluation,
        "blink_rate": round(blinks_per_minute, 2),
        "score": score,
        "feedback": evaluation,
    }
