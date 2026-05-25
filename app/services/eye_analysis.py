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

EAR_THRESHOLD = 0.21
EAR_CONSEC_FRAMES = 2


def _calculate_ear(eye_points: List[Tuple[float, float]]) -> float:
    """
    Eye Aspect Ratio 계산

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    p1, p2, p3, p4, p5, p6 = [np.array(p) for p in eye_points]

    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)

    if horizontal == 0:
        return 0.0

    return float((vertical_1 + vertical_2) / (2.0 * horizontal))


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
                "x": face_landmarks[idx].x,
                "y": face_landmarks[idx].y,
                "z": face_landmarks[idx].z,
                "pixel_x": int(face_landmarks[idx].x * width),
                "pixel_y": int(face_landmarks[idx].y * height),
            }
            for idx in LEFT_EYE_INDEXES
        ]

        right_eye = [
            {
                "index": idx,
                "x": face_landmarks[idx].x,
                "y": face_landmarks[idx].y,
                "z": face_landmarks[idx].z,
                "pixel_x": int(face_landmarks[idx].x * width),
                "pixel_y": int(face_landmarks[idx].y * height),
            }
            for idx in RIGHT_EYE_INDEXES
        ]

    return {
        "face_detected": True,
        "image_width": width,
        "image_height": height,
        "left_eye_landmarks": left_eye,
        "right_eye_landmarks": right_eye,
    }


def analyze_blinks_from_video(video_path: str) -> dict:
    """
    비디오 파일을 프레임 단위로 처리하여 EAR 기반 눈 깜빡임을 분석한다.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("비디오 파일을 열 수 없습니다.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = total_frames / fps if fps > 0 else 0.0

    mp_face_mesh = mp.solutions.face_mesh

    blink_count = 0
    consec_closed_frames = 0
    ear_history: List[float] = []
    blink_timestamps: List[float] = []
    face_detected_frames = 0
    frame_index = 0

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

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)

                if results.multi_face_landmarks:
                    face_detected_frames += 1
                    landmarks = results.multi_face_landmarks[0].landmark

                    left_eye_points = [
                        (landmarks[i].x, landmarks[i].y)
                        for i in LEFT_EYE_EAR_INDEXES
                    ]

                    right_eye_points = [
                        (landmarks[i].x, landmarks[i].y)
                        for i in RIGHT_EYE_EAR_INDEXES
                    ]

                    left_ear = _calculate_ear(left_eye_points)
                    right_ear = _calculate_ear(right_eye_points)
                    avg_ear = (left_ear + right_ear) / 2.0

                    ear_history.append(avg_ear)

                    if avg_ear < EAR_THRESHOLD:
                        consec_closed_frames += 1
                    else:
                        if consec_closed_frames >= EAR_CONSEC_FRAMES:
                            blink_count += 1
                            timestamp = frame_index / fps if fps > 0 else 0.0
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

    average_ear = float(np.mean(ear_history)) if ear_history else 0.0

    blinks_per_minute = (
        (blink_count / duration_seconds) * 60
        if duration_seconds > 0
        else 0.0
    )

    if blinks_per_minute < 8:
        evaluation = "눈 깜빡임 빈도가 낮습니다. 긴장하여 눈을 잘 깜빡이지 않는 것으로 보입니다."
    elif blinks_per_minute > 30:
        evaluation = "눈 깜빡임 빈도가 높습니다. 긴장도가 높을 수 있습니다."
    else:
        evaluation = "눈 깜빡임 빈도가 정상 범위입니다."

    return {
        "total_frames": total_frames,
        "fps": round(fps, 2),
        "duration_seconds": round(duration_seconds, 2),
        "face_detected_frames": face_detected_frames,
        "blink_count": blink_count,
        "blinks_per_minute": round(blinks_per_minute, 2),
        "average_ear": round(average_ear, 4),
        "blink_timestamps": blink_timestamps,
        "evaluation": evaluation,
    }