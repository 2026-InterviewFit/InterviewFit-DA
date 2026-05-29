from app.common.exceptions.custom_exception import CustomException
from app.common.exceptions.error_code import ErrorCode
from app.common.exceptions.status_code import StatusCode
from app.pose.services.pose_analyzer import analyze_posture
from app.pose.infrastructure.pose_detector import MediaPipeDetector
from app.pose.metrics.pose_metrics import calculate_shoulder_tilt, calculate_face_direction

def process_video(frames, fps):
    detector = MediaPipeDetector()

    try:
        results = []

        for i in range(0, len(frames), fps):
            result = process_posture(
                frame=frames[i],
                timestamp=i // fps,
                detector=detector
            )
            results.append(result)

        return results
    finally:
        detector.close()

def process_posture(frame, timestamp, detector):
    landmarks = detector.extract_landmarks(frame)

    shoulder_tilt = calculate_shoulder_tilt(
        landmarks["pose_landmarks"]
    )

    face_direction = calculate_face_direction(
        landmarks["face_landmarks"]
    )

    result = analyze_posture(
        shoulder_tilt=shoulder_tilt,
        face_direction=face_direction,
        timestamp=timestamp
    )

    return result