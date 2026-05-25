import math

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

NOSE = 1
LEFT_EYE = 33
RIGHT_EYE = 263

def calculate_shoulder_tilt(pose_landmarks):
    """
    어깨 기울기 계산
    """

    if pose_landmarks is None:
        return None

    landmarks = pose_landmarks.landmark

    left = landmarks[LEFT_SHOULDER]
    right = landmarks[RIGHT_SHOULDER]

    dx = right.x - left.x
    dy = right.y - left.y

    # 기울기 각도
    angle = math.degrees(math.atan2(dy, dx))

    # 수평 기준 보정
    tilt = abs(angle)

    # 180 근처 보정
    if tilt > 90:
        tilt = 180 - tilt

    return round(tilt, 2)

def calculate_face_direction(face_landmarks):
    """
    얼굴 정면 여부 간단 판별
    """

    if not face_landmarks:
        return None

    landmarks = face_landmarks[0].landmark

    left_eye = landmarks[LEFT_EYE]
    right_eye = landmarks[RIGHT_EYE]
    nose = landmarks[NOSE]

    eye_center_x = (left_eye.x + right_eye.x) / 2

    diff = nose.x - eye_center_x

    return round(diff, 4)