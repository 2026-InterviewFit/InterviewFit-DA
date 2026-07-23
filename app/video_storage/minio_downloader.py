from pathlib import Path
from tempfile import gettempdir
from urllib.parse import urlparse

from minio import Minio
from minio.error import S3Error

from app.core.config import settings


def get_bucket_name() -> str:
    """
    MinIO bucket 이름 반환.
    설정명 차이를 고려해 BUCKET_NAME을 우선 확인하고,
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
    Spring에서 넘어온 video_url 또는 objectKey를
    MinIO objectKey 형태로 정규화한다.

    입력 예:
    - videos/abc.webm
    - interview-videos/videos/abc.webm
    - http://localhost:9000/interview-videos/videos/abc.webm
    """

    if not video_url or not video_url.strip():
        raise ValueError(
            "video_url 값이 비어 있습니다."
        )

    bucket_name = get_bucket_name()

    value = video_url.strip()

    # URL 형태인 경우 path 부분만 추출
    if (
        value.startswith("http://")
        or value.startswith("https://")
    ):
        parsed = urlparse(value)
        value = parsed.path

    # 앞쪽 / 제거
    value = value.lstrip("/")

    # bucket 이름이 포함되어 있으면 제거
    bucket_prefix = f"{bucket_name}/"

    if value.startswith(bucket_prefix):
        value = value[len(bucket_prefix):]

    if not value:
        raise ValueError(
            "MinIO objectKey를 추출할 수 없습니다."
        )

    return value


def create_minio_client() -> Minio:
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
    MinIO에서 영상을 임시 디렉터리로 다운로드하고
    로컬 파일 경로를 반환한다.
    """

    bucket_name = get_bucket_name()

    object_key = (
        normalize_video_url_to_object_key(
            video_url
        )
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

    # objectKey 마지막 파일명 사용
    local_path = (
        temp_dir
        / Path(object_key).name
    )

    try:
        client.fget_object(
            bucket_name,
            object_key,
            str(local_path),
        )

        return str(local_path)

    except S3Error as error:

        if error.code == "NoSuchKey":
            raise FileNotFoundError(
                "MinIO에서 영상을 찾을 수 없습니다. "
                f"bucket={bucket_name}, "
                f"object_key={object_key}"
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
    로컬 파일 경로라면 그대로 사용하고,
    아니면 MinIO objectKey로 간주해 다운로드한다.
    """

    if not video_url or not video_url.strip():
        raise ValueError(
            "video_url 값이 필요합니다."
        )

    local_path = Path(
        video_url
    )

    if local_path.exists():
        return str(
            local_path.resolve()
        )

    return download_video_from_minio(
        video_url
    )