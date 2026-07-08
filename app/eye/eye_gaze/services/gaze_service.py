import os

import cv2
import mediapipe as mp

from app.eye.eye_gaze.utils.gaze_calculator import analyze_gaze_from_landmarks


class GazeService:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def analyze_frame(self, frame):
        image_height, image_width, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return {
                "face_detected": False,
                "direction": "NO_FACE",
                "avg_ratio": None,
            }

        landmarks = results.multi_face_landmarks[0].landmark

        gaze_result = analyze_gaze_from_landmarks(
            landmarks,
            image_width,
            image_height,
        )

        return {
            "face_detected": True,
            **gaze_result,
        }

    def analyze_video(self, video_path: str, frame_interval: int = 30):
        if not os.path.exists(video_path):
            raise ValueError(f"영상 파일이 존재하지 않습니다: {video_path}")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"영상 파일을 열 수 없습니다: {video_path}")

        total_frames = 0
        analyzed_frames = 0

        center_count = 0
        left_count = 0
        right_count = 0
        no_face_count = 0

        frame_index = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            total_frames += 1

            if frame_index % frame_interval != 0:
                frame_index += 1
                continue

            result = self.analyze_frame(frame)
            analyzed_frames += 1

            direction = result.get("direction", "NO_FACE")

            if direction == "CENTER":
                center_count += 1
            elif direction == "LEFT":
                left_count += 1
            elif direction == "RIGHT":
                right_count += 1
            else:
                no_face_count += 1

            frame_index += 1

        cap.release()

        if analyzed_frames == 0:
            return {
                "total_frames": total_frames,
                "analyzed_frames": 0,
                "center_count": 0,
                "left_count": 0,
                "right_count": 0,
                "no_face_count": 0,
                "away_count": 0,
                "gaze_ratio": 0,
                "gaze_score": 0,
            }

        gaze_ratio = center_count / analyzed_frames
        away_count = left_count + right_count + no_face_count
        gaze_score = round(gaze_ratio * 100)

        return {
            "total_frames": total_frames,
            "analyzed_frames": analyzed_frames,
            "center_count": center_count,
            "left_count": left_count,
            "right_count": right_count,
            "no_face_count": no_face_count,
            "away_count": away_count,
            "gaze_ratio": round(gaze_ratio, 3),
            "gaze_score": gaze_score,
        }