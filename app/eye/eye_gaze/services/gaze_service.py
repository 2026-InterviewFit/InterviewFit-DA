from typing import Dict, List, Tuple

import cv2
import mediapipe as mp
import numpy as np

from app.eye.eye_blinking.services.eye_service import (
    EAR_THRESHOLD,
    LEFT_EYE_EAR_INDEXES,
    RIGHT_EYE_EAR_INDEXES,
    _calculate_ear,
    _landmark_to_pixel_point,
)


# ============================
# MediaPipe Face Mesh 랜드마크
# ============================

LEFT_IRIS_INDEXES = [474, 475, 476, 477]
RIGHT_IRIS_INDEXES = [469, 470, 471, 472]

LEFT_EYE_LEFT_CORNER = 33
LEFT_EYE_RIGHT_CORNER = 133
RIGHT_EYE_LEFT_CORNER = 362
RIGHT_EYE_RIGHT_CORNER = 263

LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374


# ============================
# 시선 분석 기준
# ============================

HORIZONTAL_LEFT_THRESHOLD = 0.35
HORIZONTAL_RIGHT_THRESHOLD = 0.65

VERTICAL_UP_THRESHOLD = 0.35
VERTICAL_DOWN_THRESHOLD = 0.65

GAZE_AWAY_CONSEC_FRAMES = 3
FRAME_INTERVAL = 1


def _to_pixel_array(
    landmark,
    width: int,
    height: int,
) -> np.ndarray:
    point = _landmark_to_pixel_point(
        landmark,
        width,
        height,
    )

    return np.array(
        point,
        dtype=np.float32,
    )


def _iris_center(
    landmarks,
    indexes: List[int],
    width: int,
    height: int,
) -> np.ndarray:
    points = [
        _to_pixel_array(
            landmarks[index],
            width,
            height,
        )
        for index in indexes
    ]

    return np.mean(
        points,
        axis=0,
    )


def _calculate_average_ear(
    landmarks,
    width: int,
    height: int,
) -> float:
    left_eye_points = [
        _landmark_to_pixel_point(
            landmarks[index],
            width,
            height,
        )
        for index in LEFT_EYE_EAR_INDEXES
    ]

    right_eye_points = [
        _landmark_to_pixel_point(
            landmarks[index],
            width,
            height,
        )
        for index in RIGHT_EYE_EAR_INDEXES
    ]

    left_ear = _calculate_ear(
        left_eye_points,
    )

    right_ear = _calculate_ear(
        right_eye_points,
    )

    return (
        left_ear + right_ear
    ) / 2.0


def _calculate_eye_ratio(
    iris_center: np.ndarray,
    left_corner: np.ndarray,
    right_corner: np.ndarray,
    top_point: np.ndarray,
    bottom_point: np.ndarray,
) -> Tuple[float, float]:
    eye_width = abs(
        right_corner[0]
        - left_corner[0]
    )

    eye_height = abs(
        bottom_point[1]
        - top_point[1]
    )

    if eye_width == 0 or eye_height == 0:
        return 0.5, 0.5

    min_x = min(
        left_corner[0],
        right_corner[0],
    )

    min_y = min(
        top_point[1],
        bottom_point[1],
    )

    x_ratio = (
        iris_center[0] - min_x
    ) / eye_width

    y_ratio = (
        iris_center[1] - min_y
    ) / eye_height

    x_ratio = float(
        np.clip(
            x_ratio,
            0.0,
            1.0,
        )
    )

    y_ratio = float(
        np.clip(
            y_ratio,
            0.0,
            1.0,
        )
    )

    return x_ratio, y_ratio


def _classify_gaze(
    horizontal_ratio: float,
    vertical_ratio: float,
) -> str:
    if vertical_ratio < VERTICAL_UP_THRESHOLD:
        return "UP"

    if vertical_ratio > VERTICAL_DOWN_THRESHOLD:
        return "DOWN"

    if horizontal_ratio < HORIZONTAL_LEFT_THRESHOLD:
        return "LEFT"

    if horizontal_ratio > HORIZONTAL_RIGHT_THRESHOLD:
        return "RIGHT"

    return "CENTER"


def _calculate_score(
    center_ratio: float,
    face_detection_ratio: float,
) -> float:
    center_score = center_ratio * 80
    detection_score = face_detection_ratio * 20

    score = (
        center_score
        + detection_score
    )

    return round(
        min(
            100.0,
            max(
                0.0,
                score,
            ),
        ),
        2,
    )


def _build_evaluation(
    center_ratio: float,
    face_detection_ratio: float,
) -> str:
    if face_detection_ratio < 0.5:
        return (
            "얼굴이 화면에서 자주 벗어났습니다. "
            "카메라 중앙에 얼굴이 유지되도록 자세를 조정해야 합니다."
        )

    if center_ratio >= 0.8:
        return (
            "화면과 카메라를 안정적으로 응시하고 있습니다."
        )

    if center_ratio >= 0.6:
        return (
            "전반적으로 화면을 응시하고 있으나 "
            "일부 시선 이탈이 발생했습니다."
        )

    return (
        "시선 이탈이 자주 발생했습니다. "
        "답변 중 카메라를 지속적으로 바라보는 연습이 필요합니다."
    )


def analyze_gaze_from_video(
    video_path: str,
) -> Dict:
    cap = cv2.VideoCapture(
        video_path,
    )

    if not cap.isOpened():
        raise ValueError(
            "비디오 파일을 열 수 없습니다."
        )

    original_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    metadata_total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if (
        original_fps is None
        or original_fps <= 0
        or original_fps > 240
    ):
        fps = 30.0
    else:
        fps = float(
            original_fps
        )

    frame_index = 0
    analyzed_frames = 0

    face_detected_frames = 0
    no_face_frames = 0
    closed_eye_frames = 0
    gaze_detected_frames = 0

    direction_counts = {
        "CENTER": 0,
        "LEFT": 0,
        "RIGHT": 0,
        "UP": 0,
        "DOWN": 0,
    }

    gaze_away_count = 0
    gaze_away_timestamps = []

    consecutive_away_frames = 0
    is_away_event = False
    current_away_direction = None

    ear_history: List[float] = []
    horizontal_ratio_history: List[float] = []
    vertical_ratio_history: List[float] = []

    mp_face_mesh = mp.solutions.face_mesh

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

                current_frame_index = frame_index
                frame_index += 1

                if (
                    current_frame_index
                    % FRAME_INTERVAL
                    != 0
                ):
                    continue

                analyzed_frames += 1

                height, width, _ = frame.shape

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                results = face_mesh.process(
                    rgb_frame
                )

                if not results.multi_face_landmarks:
                    no_face_frames += 1

                    consecutive_away_frames = 0
                    is_away_event = False
                    current_away_direction = None

                    continue

                face_detected_frames += 1

                landmarks = (
                    results
                    .multi_face_landmarks[0]
                    .landmark
                )

                average_ear = (
                    _calculate_average_ear(
                        landmarks,
                        width,
                        height,
                    )
                )

                ear_history.append(
                    average_ear
                )

                if average_ear < EAR_THRESHOLD:
                    closed_eye_frames += 1

                    consecutive_away_frames = 0
                    is_away_event = False
                    current_away_direction = None

                    continue

                left_iris = _iris_center(
                    landmarks,
                    LEFT_IRIS_INDEXES,
                    width,
                    height,
                )

                right_iris = _iris_center(
                    landmarks,
                    RIGHT_IRIS_INDEXES,
                    width,
                    height,
                )

                left_horizontal, left_vertical = (
                    _calculate_eye_ratio(
                        left_iris,
                        _to_pixel_array(
                            landmarks[
                                LEFT_EYE_LEFT_CORNER
                            ],
                            width,
                            height,
                        ),
                        _to_pixel_array(
                            landmarks[
                                LEFT_EYE_RIGHT_CORNER
                            ],
                            width,
                            height,
                        ),
                        _to_pixel_array(
                            landmarks[
                                LEFT_EYE_TOP
                            ],
                            width,
                            height,
                        ),
                        _to_pixel_array(
                            landmarks[
                                LEFT_EYE_BOTTOM
                            ],
                            width,
                            height,
                        ),
                    )
                )

                right_horizontal, right_vertical = (
                    _calculate_eye_ratio(
                        right_iris,
                        _to_pixel_array(
                            landmarks[
                                RIGHT_EYE_LEFT_CORNER
                            ],
                            width,
                            height,
                        ),
                        _to_pixel_array(
                            landmarks[
                                RIGHT_EYE_RIGHT_CORNER
                            ],
                            width,
                            height,
                        ),
                        _to_pixel_array(
                            landmarks[
                                RIGHT_EYE_TOP
                            ],
                            width,
                            height,
                        ),
                        _to_pixel_array(
                            landmarks[
                                RIGHT_EYE_BOTTOM
                            ],
                            width,
                            height,
                        ),
                    )
                )

                horizontal_ratio = (
                    left_horizontal
                    + right_horizontal
                ) / 2.0

                vertical_ratio = (
                    left_vertical
                    + right_vertical
                ) / 2.0

                horizontal_ratio_history.append(
                    horizontal_ratio
                )

                vertical_ratio_history.append(
                    vertical_ratio
                )

                direction = _classify_gaze(
                    horizontal_ratio,
                    vertical_ratio,
                )

                direction_counts[
                    direction
                ] += 1

                gaze_detected_frames += 1

                if direction != "CENTER":
                    consecutive_away_frames += 1

                    if current_away_direction is None:
                        current_away_direction = (
                            direction
                        )

                    if (
                        direction
                        != current_away_direction
                    ):
                        current_away_direction = (
                            direction
                        )

                    if (
                        consecutive_away_frames
                        >= GAZE_AWAY_CONSEC_FRAMES
                        and not is_away_event
                    ):
                        gaze_away_count += 1
                        is_away_event = True

                        event_start_frame = (
                            current_frame_index
                            - (
                                GAZE_AWAY_CONSEC_FRAMES
                                - 1
                            )
                            * FRAME_INTERVAL
                        )

                        timestamp = (
                            event_start_frame / fps
                            if fps > 0
                            else 0.0
                        )

                        gaze_away_timestamps.append(
                            {
                                "timestamp": round(
                                    max(
                                        0.0,
                                        timestamp,
                                    ),
                                    2,
                                ),
                                "direction": (
                                    current_away_direction
                                ),
                            }
                        )

                else:
                    consecutive_away_frames = 0
                    is_away_event = False
                    current_away_direction = None

    finally:
        cap.release()

    if frame_index == 0:
        raise ValueError(
            "분석할 수 있는 프레임이 없습니다."
        )

    duration_seconds = (
        frame_index / fps
        if fps > 0
        else 0.0
    )

    center_frames = (
        direction_counts["CENTER"]
    )

    center_ratio = (
        center_frames
        / gaze_detected_frames
        if gaze_detected_frames > 0
        else 0.0
    )

    gaze_away_ratio = (
        1.0 - center_ratio
        if gaze_detected_frames > 0
        else 0.0
    )

    face_detection_ratio = (
        face_detected_frames
        / analyzed_frames
        if analyzed_frames > 0
        else 0.0
    )

    average_ear = (
        float(
            np.mean(
                ear_history
            )
        )
        if ear_history
        else 0.0
    )

    average_horizontal_ratio = (
        float(
            np.mean(
                horizontal_ratio_history
            )
        )
        if horizontal_ratio_history
        else 0.0
    )

    average_vertical_ratio = (
        float(
            np.mean(
                vertical_ratio_history
            )
        )
        if vertical_ratio_history
        else 0.0
    )

    score = _calculate_score(
        center_ratio,
        face_detection_ratio,
    )

    evaluation = _build_evaluation(
        center_ratio,
        face_detection_ratio,
    )

    return {
        "total_frames": frame_index,
        "fps": round(
            fps,
            2,
        ),
        "duration_seconds": round(
            duration_seconds,
            2,
        ),

        "face_detected_frames": (
            face_detected_frames
        ),
        "gaze_detected_frames": (
            gaze_detected_frames
        ),

        "center_frames": (
            center_frames
        ),
        "left_frames": (
            direction_counts["LEFT"]
        ),
        "right_frames": (
            direction_counts["RIGHT"]
        ),
        "up_frames": (
            direction_counts["UP"]
        ),
        "down_frames": (
            direction_counts["DOWN"]
        ),

        "center_ratio": round(
            center_ratio * 100,
            2,
        ),
        "gaze_away_ratio": round(
            gaze_away_ratio * 100,
            2,
        ),
        "gaze_away_count": (
            gaze_away_count
        ),

        "score": score,
        "evaluation": evaluation,

        "gaze_away_timestamps": (
            gaze_away_timestamps
        ),

        "analyzed_frames": (
            analyzed_frames
        ),
        "no_face_frames": (
            no_face_frames
        ),
        "closed_eye_frames": (
            closed_eye_frames
        ),
        "face_detection_ratio": round(
            face_detection_ratio * 100,
            2,
        ),

        "average_ear": round(
            average_ear,
            4,
        ),
        "average_horizontal_ratio": round(
            average_horizontal_ratio,
            4,
        ),
        "average_vertical_ratio": round(
            average_vertical_ratio,
            4,
        ),

        "metadata_total_frames": (
            metadata_total_frames
        ),
        "frame_interval": (
            FRAME_INTERVAL
        ),
        "ear_threshold": (
            EAR_THRESHOLD
        ),
        "gaze_away_consec_frames": (
            GAZE_AWAY_CONSEC_FRAMES
        ),
    }