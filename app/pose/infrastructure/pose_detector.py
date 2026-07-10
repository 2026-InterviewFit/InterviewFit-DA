import cv2
import mediapipe as mp

class MediaPipeDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_face_mesh = mp.solutions.face_mesh

        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            smooth_landmarks=False,
        )

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

    def extract_landmarks(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pose_result = self.pose.process(rgb)
        face_result = self.face_mesh.process(rgb)

        return {
            "pose_landmarks": getattr(pose_result, "pose_landmarks", None),
            "face_landmarks": getattr(face_result, "multi_face_landmarks", None)
        }

    def close(self):
        self.pose.close()
        self.face_mesh.close()