# app/video_storage/minio_downloader.py

from pathlib import Path
from tempfile import gettempdir
from urllib.parse import urlparse

from minio import Minio
from minio.error import S3Error

from app.core.config import settings


def get_bucket_name() -> str:
    """
    MinIO bucket 이름을 반환한다.

    프로젝트 설정명 차이를 고려해서
    BUCKET_NAME을 먼저 확인하고,
    없으면 MINIO_BUCKET을 사용한다.
    """

    bucket_name = (
        getattr(
            settings,
            "BUCKET_NAME",
            None,
        )
        or getattr(
            settings,
            "MINIO_BUCKET",
            None,
        )
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
    MinIO에서 실제로 사용하는 objectKey로 변환한다.

    현재 Spring 저장 구조:

    bucket:
        videos

    objectKey:
        videos/uuid_interview.webm

    중요:
    Spring에서 순수 objectKey인

        videos/abc.webm

    값이 전달되면 앞의 videos/를 제거하면 안 된다.

    전체 MinIO URL이 들어온 경우에만
    URL path의 첫 번째 bucket 이름을 제거한다.

    예:

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

        # 첫 번째 videos/는 bucket 이름이므로 제거
        #
        # videos/videos/abc.webm
        # -> videos/abc.webm
        bucket_prefix = f"{bucket_name}/"

        if value.startswith(bucket_prefix):
            value = value[len(bucket_prefix):]

    else:
        # 순수 objectKey일 경우에는
        # videos/ 접두사를 유지한다.
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

    endpoint = getattr(
        settings,
        "MINIO_ENDPOINT",
        None,
    )

    access_key = getattr(
        settings,
        "MINIO_ACCESS_KEY",
        None,
    )

    secret_key = getattr(
        settings,
        "MINIO_SECRET_KEY",
        None,
    )

    secure = getattr(
        settings,
        "MINIO_SECURE",
        False,
    )

    if not endpoint:
        raise RuntimeError(
            "MINIO_ENDPOINT 설정이 없습니다."
        )

    if not access_key:
        raise RuntimeError(
            "MINIO_ACCESS_KEY 설정이 없습니다."
        )

    if not secret_key:
        raise RuntimeError(
            "MINIO_SECRET_KEY 설정이 없습니다."
        )

    return Minio(
        endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure,
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

    # objectKey 마지막 파일명을
    # 임시 로컬 파일명으로 사용한다.
    filename = Path(object_key).name

    local_path = (
        temp_dir
        / filename
    )

    print(
        "======================================"
    )
    print(
        "[MinIO 영상 다운로드 요청]"
    )
    print(
        f"원본 videoUrl = {video_url}"
    )
    print(
        f"bucket = {bucket_name}"
    )
    print(
        f"objectKey = {object_key}"
    )
    print(
        f"localPath = {local_path}"
    )
    print(
        "======================================"
    )

    try:
        client.fget_object(
            bucket_name,
            object_key,
            str(local_path),
        )

        print(
            "======================================"
        )
        print(
            "[MinIO 영상 다운로드 성공]"
        )
        print(
            f"bucket = {bucket_name}"
        )
        print(
            f"objectKey = {object_key}"
        )
        print(
            f"localPath = {local_path}"
        )
        print(
            "======================================"
        )

        return str(local_path)

    except S3Error as error:
        print(
            "======================================"
        )
        print(
            "[MinIO 영상 다운로드 실패]"
        )
        print(
            f"bucket = {bucket_name}"
        )
        print(
            f"objectKey = {object_key}"
        )
        print(
            f"errorCode = {error.code}"
        )
        print(
            f"error = {error}"
        )
        print(
            "======================================"
        )

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
    video_url이 실제 로컬 파일 경로이면 그대로 사용한다.

    로컬 파일이 아니면 MinIO objectKey 또는
    MinIO URL로 판단해서 다운로드한다.
    """

    if not video_url or not video_url.strip():
        raise ValueError(
            "video_url 값이 필요합니다."
        )

    value = video_url.strip()

    local_path = Path(value)

    if local_path.exists():
        resolved_path = local_path.resolve()

        print(
            "======================================"
        )
        print(
            "[로컬 영상 사용]"
        )
        print(
            f"path = {resolved_path}"
        )
        print(
            "======================================"
        )

        return str(resolved_path)

    return download_video_from_minio(
        value
    )