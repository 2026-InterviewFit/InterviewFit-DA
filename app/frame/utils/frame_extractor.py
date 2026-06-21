import os
import cv2
import uuid
import logging

logger = logging.getLogger(__name__)


def extract_frames(video_path: str, frame_interval: int = 30):

    logger.info("받은 경로: %s", video_path)
    logger.info("현재 폴더: %s", os.getcwd())
    logger.info("파일 존재 여부: %s", os.path.exists(video_path))

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logger.error("영상 파일 열기 실패: %s", video_path)
        raise Exception("영상 파일을 열 수 없습니다.")

    output_dir = os.path.join(
        "uploads",
        "frames",
        str(uuid.uuid4())
    )
    os.makedirs(output_dir, exist_ok=True)

    frame_count = 0
    extracted_count = 0
    saved_frame_paths = []

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = f"frame_{extracted_count}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)

            cv2.imwrite(frame_path, frame)

            saved_frame_paths.append(frame_path)
            extracted_count += 1

        frame_count += 1

    cap.release()

    logger.info("프레임 추출 완료: %s개", extracted_count)
    logger.info("저장 위치: %s", output_dir)

    return {
        "frameCount": extracted_count,
        "framePaths": saved_frame_paths,
        "outputDir": output_dir
    }