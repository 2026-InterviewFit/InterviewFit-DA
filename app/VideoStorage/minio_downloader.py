# ================================
# 수정 전 코드
# ================================
# from minio import Minio
# import os
# import tempfile
#
# minio_client = Minio(
#     "localhost:9000",
#     access_key="minioadmin",
#     secret_key="minioadmin123",
#     secure=False
# )
#
# BUCKET_NAME = "videos"
#
#
# def download_video_from_minio(object_key: str) -> str:
#     temp_dir = tempfile.gettempdir()
#     file_name = os.path.basename(object_key)
#     local_path = os.path.join(temp_dir, file_name)
#
#     minio_client.fget_object(
#         BUCKET_NAME,
#         object_key,
#         local_path
#     )
#
#     print("MinIO 다운로드 완료:", local_path)
#
#     return local_path


# ================================
# 수정 후 코드
# ================================
import os
import tempfile
from pathlib import Path

from minio import Minio

from app.core.config import settings


# 팀원 VideoStorage, eye-blink 분석이 공통으로 사용하는 MinIO client.
#
# 수정 이유:
# 기존 코드는 MinIO 주소, access key, secret key, bucket 이름이
# 코드 안에 직접 하드코딩되어 있었다.
#
# 로컬 실행에서는 localhost:9000이 맞지만,
# Docker 환경에서는 minio:9000처럼 주소가 달라질 수 있다.
#
# 따라서 settings/.env에서 값을 읽도록 바꾸면
# 환경이 바뀌어도 코드 수정 없이 .env만 바꾸면 된다.
minio_client = Minio(
    settings.MINIO_ENDPOINT,
    access_key=settings.MINIO_ACCESS_KEY,
    secret_key=settings.MINIO_SECRET_KEY,
    secure=settings.MINIO_SECURE,
)

BUCKET_NAME = settings.MINIO_BUCKET


def download_video_from_minio(object_key: str) -> str:
    """
    MinIO objectKey를 받아 FastAPI 서버가 읽을 수 있는 임시 로컬 파일로 다운로드한다.

    Spring VideoService가 MinIO에 저장한 뒤 반환하는 값은 보통
    `videos/uuid_filename.webm` 형태의 objectKey다.

    OpenCV(cv2.VideoCapture)는 `videos/xxx.webm` 같은 MinIO objectKey를
    직접 읽을 수 없으므로 FastAPI 내부 임시 폴더로 내려받은 뒤,
    실제 로컬 파일 경로를 반환한다.
    """

    # objectKey가 비어 있으면 MinIO 호출 전에 명확한 에러를 발생시킨다.
    # 기존 코드는 빈 값이 들어와도 바로 fget_object를 호출해서
    # 에러 원인을 파악하기 어려웠다.
    if not object_key or not object_key.strip():
        raise ValueError("MinIO objectKey가 비어 있습니다.")

    # objectKey 예시:
    # videos/uuid_interview.webm
    #
    # 실제 로컬 저장에는 파일명만 사용한다.
    # 예: uuid_interview.webm
    safe_file_name = os.path.basename(object_key) or "interview_video.webm"

    # FastAPI 서버의 임시 폴더 아래 interviewfit-videos 폴더를 만들어 저장한다.
    # 예: /tmp/interviewfit-videos/uuid_interview.webm
    temp_dir = Path(tempfile.gettempdir()) / "interviewfit-videos"
    temp_dir.mkdir(parents=True, exist_ok=True)

    local_path = str(temp_dir / safe_file_name)

    # MinIO에서 objectKey에 해당하는 파일을 local_path로 다운로드한다.
    minio_client.fget_object(
        BUCKET_NAME,
        object_key,
        local_path,
    )

    print("MinIO 다운로드 완료:", local_path)
    return local_path


def resolve_video_path(video_url: str) -> str:
    """
    Spring/FastAPI 테스트에서 들어오는 videoUrl을 실제 파일 경로로 변환한다.

    지원하는 입력값:
    1. 로컬 파일 경로
       - Swagger 단독 테스트나 /blinks/upload 내부 테스트용
       - 예: /Users/mxxxdxxx/sample.webm

    2. MinIO objectKey
       - 팀원 VideoStorage와 연결되는 실제 통합 방식
       - 예: videos/uuid_sample.webm

    추가 이유:
    기존 팀원 코드는 MinIO 다운로드만 담당했다.
    그런데 eye-blink 분석에서는 videoUrl이 로컬 경로일 수도 있고,
    MinIO objectKey일 수도 있다.

    그래서 eye-blink, pose, speech 같은 영상 분석 기능들이
    videoUrl을 받았을 때 실제 분석 가능한 파일 경로로 바꾸는 공통 함수가 필요하다.
    """

    if not video_url or not video_url.strip():
        raise ValueError("videoUrl 값이 비어 있습니다.")

    # 이미 FastAPI 서버에서 접근 가능한 로컬 파일이면 그대로 사용한다.
    #
    # 예:
    # /Users/mxxxdxxx/interview-fit/ai-server/test.mp4
    if os.path.exists(video_url):
        return video_url

    # 팀원 VideoService가 MinIO에 저장한 영상은 보통 videos/로 시작한다.
    #
    # 예:
    # videos/uuid_interview.webm
    #
    # 이 경우에는 MinIO에서 임시 파일로 다운로드한 뒤 그 경로를 반환한다.
    if video_url.startswith("videos/"):
        return download_video_from_minio(video_url)

    # 로컬 파일도 아니고 MinIO objectKey도 아니면 분석할 수 없는 값이다.
    raise FileNotFoundError(f"지원하지 않거나 찾을 수 없는 videoUrl입니다: {video_url}")