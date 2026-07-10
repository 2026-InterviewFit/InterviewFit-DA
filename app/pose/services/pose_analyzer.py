from app.pose.constants.pose_status import ShoulderStatus, FaceStatus
from app.pose.schemas.pose_schema import PoseAnalysisResponse

SHOULDER_THRESHOLD = 5
FACE_THRESHOLD = 0.1

def analyze_posture(shoulder_tilt, face_direction, timestamp):
    shoulder_status = ShoulderStatus.BAD.value
    face_status = FaceStatus.BAD.value

    if shoulder_tilt is None:
        shoulder_status = ShoulderStatus.NOT_DETECTED.value
    elif shoulder_tilt <= SHOULDER_THRESHOLD:
        shoulder_status = ShoulderStatus.GOOD.value

    if face_direction is None:
        face_status = FaceStatus.NOT_DETECTED.value
    elif abs(face_direction) <= FACE_THRESHOLD:
        face_status = FaceStatus.GOOD.value

    return PoseAnalysisResponse(
        timestamp=timestamp,
        shoulder_tilt=shoulder_tilt,
        face_direction=face_direction,
        shoulder_status=shoulder_status,
        face_status=face_status
    )
