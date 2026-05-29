import cv2


def extract_frames(video_path: str, frame_interval: int = 30):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise Exception("영상 파일을 열 수 없습니다.")

    frames = []

    frame_count = 0

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % frame_interval == 0:
            frames.append(frame)

        frame_count += 1

    cap.release()

    return frames