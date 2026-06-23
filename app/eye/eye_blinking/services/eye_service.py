from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np


# ============================
# MediaPipe Face Mesh 랜드마크 인덱스
# ============================

# 눈 전체 윤곽 랜드마크
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
# 깜빡임 감지 임계값
# ============================

# 기존 0.21은 사람/카메라에 따라 너무 낮을 수 있음
# count가 너무 안 올라가면 0.24 ~ 0.27 사이에서 조정
EAR_THRESHOLD = 0.24

# 2프레임 이상 연속으로 눈이 감겼을 때 깜빡임으로 인정
# 너무 민감하게 잡고 싶으면 1로 바꿀 수 있음
EAR_CONSEC_FRAMES = 2


def _calculate_ear(eye_points: List[Tuple[float, float]]) -> float:
    """
    Eye Aspect Ratio 계산

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    EAR 값이 낮을수록 눈을 감은 상태에 가까움.
    """

    if len(eye_points) != 6:
        raise ValueError("EAR 계산에는 6개의 눈 좌표가 필요합니다.")

    p1, p2, p3, p4, p5, p6 = [np.array(p, dtype=np.float32) for p in eye_points]

    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)

    if horizontal == 0:
        return 0.0

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return float(ear)


def _landmark_to_pixel_point(landmark, width: int, height: int) -> Tuple[float, float]:
    """
    MediaPipe 정규화 좌표를 실제 pixel 좌표로 변환.
    """

    return landmark.x * width, landmark.y * height


def extract_eye_landmarks_from_image(image_bytes: bytes) -> dict:
    """
    이미지 bytes를 받아 왼쪽 눈 / 오른쪽 눈 랜드마크 좌표를 반환한다.
    """

    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("이미지를 읽을 수 없습니다.")

    height, width, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
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


def _get_blink_evaluation(blinks_per_minute: float) -> str:
    """
    눈 깜빡임 빈도에 따른 평가 문장.
    """

    if blinks_per_minute < 8:
        return "눈 깜빡임 빈도가 낮습니다. 긴장하여 눈을 잘 깜빡이지 않는 것으로 보입니다."

    if blinks_per_minute > 30:
        return "눈 깜빡임 빈도가 높습니다. 긴장도가 높을 수 있습니다."

    return "눈 깜빡임 빈도가 정상 범위입니다."


def _get_blink_score(blinks_per_minute: float) -> int:
    """
    간단 점수화.
    15~20회/분을 가장 안정적인 범위로 보고 점수 부여.
    """

    if 15 <= blinks_per_minute <= 20:
        return 100

    if 8 <= blinks_per_minute < 15:
        return 80

    if 20 < blinks_per_minute <= 30:
        return 75

    if blinks_per_minute < 8:
        return 50

    return 40


def analyze_blinks_from_video(video_path: str) -> dict:
    """
    비디오 파일을 프레임 단위로 처리하여 EAR 기반 눈 깜빡임을 분석한다.
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("비디오 파일을 열 수 없습니다.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps is None or fps <= 0:
        fps = 30.0

    duration_seconds = total_frames / fps if total_frames > 0 else 0.0

    mp_face_mesh = mp.solutions.face_mesh

    blink_count = 0
    consec_closed_frames = 0

    ear_history: List[float] = []
    blink_timestamps: List[float] = []

    face_detected_frames = 0
    frame_index = 0

    # 디버깅용
    closed_frame_indexes: List[int] = []

    try:
        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                height, width, _ = frame.shape
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)

                if results.multi_face_landmarks:
                    face_detected_frames += 1
                    landmarks = results.multi_face_landmarks[0].landmark

                    # 핵심 수정:
                    # 기존에는 (landmarks[i].x, landmarks[i].y) 정규화 좌표로 계산했음.
                    # 이제는 width, height를 곱해서 pixel 좌표로 계산함.
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
                        closed_frame_indexes.append(frame_index)

                    else:
                        if consec_closed_frames >= EAR_CONSEC_FRAMES:
                            blink_count += 1

                            # 깜빡임이 끝난 시점 기준 timestamp
                            timestamp = frame_index / fps
                            blink_timestamps.append(round(timestamp, 2))

                        consec_closed_frames = 0

                else:
                    # 얼굴이 안 잡힌 프레임에서 직전 감김 상태 마무리
                    if consec_closed_frames >= EAR_CONSEC_FRAMES:
                        blink_count += 1
                        timestamp = frame_index / fps
                        blink_timestamps.append(round(timestamp, 2))

                    consec_closed_frames = 0

                frame_index += 1

            # 영상 마지막 프레임에서 눈을 감은 상태로 끝나는 경우도 카운트
            if consec_closed_frames >= EAR_CONSEC_FRAMES:
                blink_count += 1
                timestamp = (frame_index - 1) / fps if fps > 0 else 0.0
                blink_timestamps.append(round(timestamp, 2))

    finally:
        cap.release()

    if frame_index == 0:
        raise ValueError("분석할 수 있는 프레임이 없습니다.")

    if duration_seconds <= 0:
        duration_seconds = frame_index / fps

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

    return {
        "total_frames": total_frames,
        "fps": round(fps, 2),
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

        # 기존 응답 필드 호환용
        "blink_rate": round(blinks_per_minute, 2),
        "score": score,
        "feedback": evaluation,
    }