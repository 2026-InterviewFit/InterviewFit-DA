import cv2
from app.VideoStorage.minio_downloader import download_video_from_minio
from app.common.exceptions.custom_exception import CustomException
from app.common.exceptions.error_code import ErrorCode
from app.common.exceptions.status_code import StatusCode
from app.frame.service.frame_service import analyze_frames
from app.pose.services.pose_analyzer import analyze_posture
from app.pose.infrastructure.pose_detector import MediaPipeDetector
from app.pose.metrics.pose_metrics import calculate_shoulder_tilt, calculate_face_direction

def analyze_pose_video(video_object_key: str):
    local_video_path = download_video_from_minio(video_object_key)
    frame_result = analyze_frames(local_video_path)

    frames = []

    for path in frame_result["framePaths"]:
        frame = cv2.imread(path)
        if frame is not None:
            frames.append(frame)

    if len(frames) == 0:
        raise CustomException(
            status=StatusCode.BAD_REQUEST,
            code=ErrorCode.FRAME_NOT_FOUND,
            message="영상에서 프레임을 추출할 수 없습니다."
        )

    return process_video(frames, fps=1)

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