import math

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

NOSE = 1
LEFT_EYE = 33
RIGHT_EYE = 263

def calculate_shoulder_tilt(pose_landmarks):
    """
    어깨 기울기 계산 (degree)
    """

    if pose_landmarks is None:
        return None

    landmarks = pose_landmarks.landmark

    left = landmarks[LEFT_SHOULDER]
    right = landmarks[RIGHT_SHOULDER]

    # 어깨 라인의 각도
    angle = math.degrees(math.atan2(
        right.y - left.y,
        right.x - left.x
    ))

    # 0~90 범위로 정규화
    tilt = abs(angle)
    tilt = min(tilt, 180 - tilt)

    return round(tilt, 2)

def calculate_face_direction(face_landmarks):
    """
    얼굴 방향 분석 (yaw)
    """

    if not face_landmarks:
        return None

    landmarks = face_landmarks[0].landmark

    left_eye = landmarks[LEFT_EYE]
    right_eye = landmarks[RIGHT_EYE]
    nose = landmarks[NOSE]

    eye_center_x = (left_eye.x + right_eye.x) / 2

    face_width = abs(right_eye.x - left_eye.x)

    if face_width < 1e-6:
        return None

    # yaw (좌우)
    yaw = (nose.x - eye_center_x) / face_width

    return round(yaw, 2)