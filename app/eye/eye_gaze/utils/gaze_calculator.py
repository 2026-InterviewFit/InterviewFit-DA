from app.eye.eye_gaze.utils.gaze_constants import (
    LEFT_IRIS,
    RIGHT_IRIS,
    LEFT_EYE_LEFT,
    LEFT_EYE_RIGHT,
    RIGHT_EYE_LEFT,
    RIGHT_EYE_RIGHT,
    CENTER_MIN,
    CENTER_MAX,
)


def landmark_to_pixel(landmark, image_width: int, image_height: int):
    return int(landmark.x * image_width), int(landmark.y * image_height)


def get_iris_center(landmarks, iris_indices, image_width: int, image_height: int):
    x_sum = 0
    y_sum = 0

    for index in iris_indices:
        x, y = landmark_to_pixel(landmarks[index], image_width, image_height)
        x_sum += x
        y_sum += y

    return x_sum // len(iris_indices), y_sum // len(iris_indices)


def calculate_eye_ratio(eye_left_x: int, eye_right_x: int, iris_x: int):
    eye_width = eye_right_x - eye_left_x

    if eye_width == 0:
        return 0.5

    return (iris_x - eye_left_x) / eye_width


def get_gaze_direction(ratio: float):
    if CENTER_MIN <= ratio <= CENTER_MAX:
        return "CENTER"

    if ratio < CENTER_MIN:
        return "LEFT"

    return "RIGHT"


def analyze_gaze_from_landmarks(landmarks, image_width: int, image_height: int):
    left_eye_left_x, _ = landmark_to_pixel(
        landmarks[LEFT_EYE_LEFT], image_width, image_height
    )
    left_eye_right_x, _ = landmark_to_pixel(
        landmarks[LEFT_EYE_RIGHT], image_width, image_height
    )

    right_eye_left_x, _ = landmark_to_pixel(
        landmarks[RIGHT_EYE_LEFT], image_width, image_height
    )
    right_eye_right_x, _ = landmark_to_pixel(
        landmarks[RIGHT_EYE_RIGHT], image_width, image_height
    )

    left_iris_x, left_iris_y = get_iris_center(
        landmarks, LEFT_IRIS, image_width, image_height
    )
    right_iris_x, right_iris_y = get_iris_center(
        landmarks, RIGHT_IRIS, image_width, image_height
    )

    left_ratio = calculate_eye_ratio(
        left_eye_left_x,
        left_eye_right_x,
        left_iris_x,
    )

    right_ratio = calculate_eye_ratio(
        right_eye_left_x,
        right_eye_right_x,
        right_iris_x,
    )

    avg_ratio = (left_ratio + right_ratio) / 2
    direction = get_gaze_direction(avg_ratio)

    return {
        "left_ratio": round(left_ratio, 3),
        "right_ratio": round(right_ratio, 3),
        "avg_ratio": round(avg_ratio, 3),
        "direction": direction,
        "left_iris": {
            "x": left_iris_x,
            "y": left_iris_y,
        },
        "right_iris": {
            "x": right_iris_x,
            "y": right_iris_y,
        },
    }