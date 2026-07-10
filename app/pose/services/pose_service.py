from app.pose.metrics.pose_metrics import calculate_shoulder_tilt, calculate_face_direction
from app.pose.utils.video_utils import extract_frames, delete_temp_video
from app.video_storage.minio_downloader import resolve_video_path
from app.common.exceptions.custom_exception import CustomException
from app.common.exceptions.error_code import ErrorCode
from app.common.exceptions.status_code import StatusCode
from app.pose.services.pose_analyzer import analyze_posture
from app.pose.infrastructure.pose_detector import MediaPipeDetector

def analyze_pose_video(video_object_key: str):
    local_video_path = resolve_video_path(video_object_key)

    try:
        frames = extract_frames(local_video_path)

        if len(frames) == 0:
            raise CustomException(
                status=StatusCode.BAD_REQUEST,
                code=ErrorCode.FRAME_NOT_FOUND,
                message="영상에서 프레임을 추출할 수 없습니다."
            )

        return process_pose_video(frames)
    finally:
        delete_temp_video(local_video_path)

def process_pose_video(frames):
    detector = MediaPipeDetector()

    try:
        results = []

        for data in frames:
            result = process_posture(
                frame=data["frame"],
                timestamp=data["timestamp"],
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