# from app.common.exceptions.custom_exception import CustomException
# from app.common.exceptions.error_code import ErrorCode
# from app.common.exceptions.status_code import StatusCode
# from app.pose.detectors.pose_detector import extract_landmarks
# from app.pose.analyzers.pose_analyzer import analyze_posture
# from app.pose.metrics.pose_metrics import calculate_shoulder_tilt, calculate_face_direction
#
# def process_video(frames, fps):
#     frame_interval = fps
#     results = []
#
#     for i in range(0, len(frames), frame_interval):
#         frame = frames[i]
#
#         result = process_posture(
#             frame=frame,
#             timestamp=i // fps
#         )
#
#         results.append(result)
#
#     return results
#
# def process_posture(frame, timestamp):
#     if frame is None:
#         raise CustomException(
#             status=StatusCode.BAD_REQUEST,
#             code=ErrorCode.IMAGE_READ_FAILED,
#             message="이미지를 읽을 수 없습니다."
#         )
#
#     landmarks = extract_landmarks(frame)
#
#     shoulder_tilt = calculate_shoulder_tilt(
#         landmarks["pose_landmarks"]
#     )
#
#     face_direction = calculate_face_direction(
#         landmarks["face_landmarks"]
#     )
#
#     print(shoulder_tilt)
#     print(face_direction)
#
#     result = analyze_posture(
#         shoulder_tilt=shoulder_tilt,
#         face_direction=face_direction,
#         timestamp=timestamp
#     )
#
#     return result