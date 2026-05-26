import cv2
import mediapipe as mp

class MediaPipeDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_face_mesh = mp.solutions.face_mesh

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
        )

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
        )

    def extract_landmarks(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pose_result = self.pose.process(rgb)
        face_result = self.face_mesh.process(rgb)

        return {
            "pose_landmarks": pose_result.pose_landmarks,
            "face_landmarks": face_result.multi_face_landmarks
        }

    def close(self):
        self.pose.close()
        self.face_mesh.close()