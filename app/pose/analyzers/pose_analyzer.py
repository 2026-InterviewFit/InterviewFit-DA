from app.pose.constants.pose_status import ShoulderStatus, FaceStatus
from app.pose.schemas.pose_schema import PoseAnalysisResponse

SHOULDER_THRESHOLD = 5
FACE_THRESHOLD = 0.03

def analyze_posture(shoulder_tilt, face_direction, timestamp):
    shoulder_status = str(ShoulderStatus.BAD)
    face_status = str(FaceStatus.BAD)

    if shoulder_tilt is not None and abs(shoulder_tilt) <= SHOULDER_THRESHOLD:
        shoulder_status = str(ShoulderStatus.GOOD)

    if face_direction is not None and abs(face_direction) <= FACE_THRESHOLD:
        face_status = str(FaceStatus.GOOD)

    return PoseAnalysisResponse(
        timestamp=timestamp,
        shoulder_tilt=shoulder_tilt,
        face_direction=face_direction,
        shoulder_status=shoulder_status,
        face_status=face_status
    )
    #     {
    #     "timestamp": timestamp,
    #     "shoulder_tilt": shoulder_tilt,
    #     "face_direction": face_direction,
    #     "shoulder_status": shoulder_status,
    #     "face_status": face_status
    # }