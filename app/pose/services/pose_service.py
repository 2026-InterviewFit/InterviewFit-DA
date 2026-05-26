from app.common.exceptions.custom_exception import CustomException
from app.common.exceptions.error_code import ErrorCode
from app.common.exceptions.status_code import StatusCode
from app.pose.analyzers.pose_analyzer import analyze_posture
from app.pose.detectors.pose_detector import MediaPipeDetector
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

            # landmark 감지 실패 프레임 스킵
            if result is not None:
                results.append(result)

        # 전체 프레임 실패
        if len(results) == 0:
            raise CustomException(
                status=StatusCode.BAD_REQUEST,
                code=ErrorCode.POSE_NOT_DETECTED,
                message="자세를 감지할 수 없습니다."
            )

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

    if shoulder_tilt is None or face_direction is None:
        return None

    result = analyze_posture(
        shoulder_tilt=shoulder_tilt,
        face_direction=face_direction,
        timestamp=timestamp
    )

    return result