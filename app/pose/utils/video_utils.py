from pathlib import Path
import logging
import cv2

from app.common.exceptions.custom_exception import CustomException
from app.common.exceptions.error_code import ErrorCode
from app.common.exceptions.status_code import StatusCode

logger = logging.getLogger(__name__)

def extract_frames(video_path: str):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise CustomException(
            status=StatusCode.BAD_REQUEST,
            code=ErrorCode.VIDEO_OPEN_FAILED,
            message="영상을 열 수 없습니다."
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        cap.release()
        raise CustomException(
            status=StatusCode.BAD_REQUEST,
            code=ErrorCode.INVALID_VIDEO_FPS,
            message="영상 FPS 정보를 가져올 수 없습니다."
        )

    frames = []

    frame_index = 0
    interval = round(fps)

    last_frame = None
    last_timestamp = 0.0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        timestamp = frame_index / fps

        # 마지막 프레임 저장
        last_frame = frame
        last_timestamp = timestamp

        # 1초 단위 추출
        if frame_index % interval == 0:
            frames.append({
                "frame": frame,
                "timestamp": frame_index // interval
            })

        frame_index += 1

    cap.release()

    # 마지막 프레임이 이미 추가되지 않은 경우 추가
    if last_frame is not None and (not frames or last_timestamp > frames[-1]["timestamp"]):
        frames.append({
            "frame": last_frame,
            "timestamp": last_timestamp
        })

    return frames

def delete_temp_video(video_path: str) -> None:
    path = Path(video_path)

    if not path.exists():
        return

    try:
        path.unlink()
    except OSError as e:
        logger.warning("임시 영상 삭제 실패: %s, error=%s", video_path, e)
