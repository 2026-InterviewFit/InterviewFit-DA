import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
)

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
)

def extract_landmarks(frame):
    """
    frame: OpenCV BGR image
    """

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pose_result = pose.process(rgb_frame)
    face_result = face_mesh.process(rgb_frame)

    return {
        "pose_landmarks": pose_result.pose_landmarks,
        "face_landmarks": face_result.multi_face_landmarks
    }