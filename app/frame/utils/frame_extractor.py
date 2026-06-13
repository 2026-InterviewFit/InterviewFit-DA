import os
import cv2
import uuid


def extract_frames(video_path: str, frame_interval: int = 30):

    print("받은 경로:", video_path)
    print("현재 폴더:", os.getcwd())
    print("파일 존재:", os.path.exists(video_path))

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise Exception("영상 파일을 열 수 없습니다.")

    # 저장 폴더 생성
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

    return {
        "frameCount": extracted_count,
        "framePaths": saved_frame_paths,
        "outputDir": output_dir
    }