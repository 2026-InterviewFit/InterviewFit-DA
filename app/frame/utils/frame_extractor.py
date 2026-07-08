import os
import uuid
import cv2

DEFAULT_FRAME_INTERVAL = 30
FRAME_OUTPUT_DIR = os.path.join("uploads", "frames")


def extract_frames(
    video_path: str,
    frame_interval: int = DEFAULT_FRAME_INTERVAL
) -> dict:

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다. : {video_path}")

    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        raise ValueError("영상 파일을 열 수 없습니다.")

    output_dir = create_output_directory()

    frame_index = 0
    extracted_count = 0
    frame_paths = []

    while video_capture.isOpened():
        success, frame = video_capture.read()

        if not success:
            break

        if frame_index % frame_interval == 0:
            frame_path = save_frame(
                frame,
                output_dir,
                extracted_count
            )

            frame_paths.append(frame_path)
            extracted_count += 1

        frame_index += 1

    video_capture.release()

    return {
        "frameCount": extracted_count,
        "framePaths": frame_paths,
        "outputDir": output_dir
    }


def create_output_directory() -> str:
    output_dir = os.path.join(
        FRAME_OUTPUT_DIR,
        str(uuid.uuid4())
    )

    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_frame(frame, output_dir: str, index: int) -> str:
    frame_path = os.path.join(
        output_dir,
        f"frame_{index}.jpg"
    )

    cv2.imwrite(frame_path, frame)
    return frame_path