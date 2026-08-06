from pathlib import Path
from tempfile import gettempdir
from urllib.parse import urlparse

from minio import Minio
from minio.error import S3Error

from app.core.config import settings


def get_bucket_name() -> str:
    """
    MinIO bucket 이름을 반환한다.

    설정명 차이를 고려해 BUCKET_NAME을 먼저 확인하고,
    없으면 MINIO_BUCKET을 사용한다.
    """
    bucket_name = (
        getattr(settings, "BUCKET_NAME", None)
        or getattr(settings, "MINIO_BUCKET", None)
    )

    if not bucket_name:
        raise RuntimeError(
            "MinIO bucket 설정이 없습니다."
        )

    return bucket_name


def normalize_video_url_to_object_key(
    video_url: str,
) -> str:
    """
    Spring에서 전달한 videoUrl 또는 objectKey를
    MinIO objectKey 형태로 변환한다.

    Spring에서 전달되는 실제 값:

        videos/uuid_interview.webm

    여기서 앞의 videos/는 bucket 이름이 아니라
    실제 objectKey에 포함된 폴더이므로 삭제하지 않는다.

    전체 URL이 전달된 경우에만 URL 경로의 첫 번째
    bucket 이름을 제거한다.

    예시:

        videos/abc.webm
        -> videos/abc.webm

        http://localhost:9000/videos/videos/abc.webm
        -> videos/abc.webm
    """
    if not video_url or not video_url.strip():
        raise ValueError(
            "video_url 값이 비어 있습니다."
        )

    bucket_name = get_bucket_name()
    value = video_url.strip()

    is_full_url = (
        value.startswith("http://")
        or value.startswith("https://")
    )

    if is_full_url:
        parsed = urlparse(value)

        # /videos/videos/abc.webm
        # -> videos/videos/abc.webm
        value = parsed.path.lstrip("/")

        # 전체 URL에서는 첫 번째 경로가 bucket 이름이다.
        #
        # videos/videos/abc.webm
        # -> videos/abc.webm
        bucket_prefix = f"{bucket_name}/"

        if value.startswith(bucket_prefix):
            value = value[len(bucket_prefix):]

    else:
        # Spring에서 순수 objectKey가 전달된 경우에는
        # videos/ 접두사를 제거하지 않는다.
        #
        # videos/abc.webm
        # -> videos/abc.webm
        value = value.lstrip("/")

    if not value:
        raise ValueError(
            "MinIO objectKey를 추출할 수 없습니다."
        )

    return value


def create_minio_client() -> Minio:
    """
    프로젝트 설정값으로 MinIO 클라이언트를 생성한다.
    """
    return Minio(
        settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=settings.MINIO_SECURE,
    )


def download_video_from_minio(
    video_url: str,
) -> str:
    """
    MinIO에서 영상을 임시 폴더로 다운로드하고
    다운로드된 로컬 파일 경로를 반환한다.
    """
    bucket_name = get_bucket_name()

    object_key = normalize_video_url_to_object_key(
        video_url
    )

    client = create_minio_client()

    temp_dir = (
        Path(gettempdir())
        / "interviewfit-videos"
    )

    temp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # objectKey의 마지막 파일명을 임시 파일명으로 사용한다.
    local_path = (
        temp_dir
        / Path(object_key).name
    )

    print("======================================")
    print("[MinIO 영상 다운로드 요청]")
    print(f"원본 videoUrl = {video_url}")
    print(f"bucket = {bucket_name}")
    print(f"objectKey = {object_key}")
    print(f"localPath = {local_path}")
    print("======================================")

    try:
        client.fget_object(
            bucket_name,
            object_key,
            str(local_path),
        )

        print("======================================")
        print("[MinIO 영상 다운로드 성공]")
        print(f"bucket = {bucket_name}")
        print(f"objectKey = {object_key}")
        print(f"localPath = {local_path}")
        print("======================================")

        return str(local_path)

    except S3Error as error:
        if error.code in {
            "NoSuchKey",
            "NoSuchObject",
            "NoSuchBucket",
        }:
            raise FileNotFoundError(
                "MinIO에서 영상을 찾을 수 없습니다. "
                f"bucket={bucket_name}, "
                f"object_key={object_key}, "
                f"error_code={error.code}"
            ) from error

        raise RuntimeError(
            "MinIO 영상 다운로드 중 오류가 발생했습니다. "
            f"bucket={bucket_name}, "
            f"object_key={object_key}, "
            f"error={error}"
        ) from error


def resolve_video_path(
    video_url: str,
) -> str:
    """
    video_url이 로컬 파일 경로이면 그대로 사용한다.

    로컬 파일이 아니면 MinIO objectKey 또는
    MinIO URL로 판단해 다운로드한다.
    """
    if not video_url or not video_url.strip():
        raise ValueError(
            "video_url 값이 필요합니다."
        )

    local_path = Path(video_url)

    if local_path.exists():
        return str(
            local_path.resolve()
        )

    return download_video_from_minio(
        video_url
    )