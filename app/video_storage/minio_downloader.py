from pathlib import Path
from tempfile import gettempdir
from urllib.parse import urlparse

from minio import Minio
from minio.error import S3Error

from app.core.config import settings


def get_bucket_name() -> str:
    """
    팀원마다 설정 이름이 다를 수 있어서 둘 다 대응.
    settings에 BUCKET_NAME이 있으면 그걸 우선 사용하고,
    없으면 MINIO_BUCKET을 사용한다.
    """
    return getattr(settings, "BUCKET_NAME", None) or getattr(settings, "MINIO_BUCKET")


def normalize_video_url_to_object_key(video_url: str) -> str:
    """
    Spring에서 넘어올 수 있는 값 예시:
    - videos/abc.webm
    - interview-videos/abc.webm
    - abc.webm
    - http://localhost:9000/videos/abc.webm

    Python MinIO 클라이언트는 bucket_name과 object_key를 따로 받기 때문에
    videoUrl에 bucket 이름이 붙어 있으면 제거해야 한다.
    """
    bucket_name = get_bucket_name()

    value = video_url.strip()

    # URL 형태로 넘어온 경우 path만 뽑기
    if value.startswith("http://") or value.startswith("https://"):
        parsed = urlparse(value)
        value = parsed.path.lstrip("/")

    bucket_prefix = f"{bucket_name}/"

    if value.startswith(bucket_prefix):
        value = value[len(bucket_prefix):]

    return value


def build_candidate_object_keys(video_url: str) -> list[str]:
    """
    안전장치:
    1순위: bucket prefix 제거한 objectKey
    2순위: 원본 videoUrl에서 앞 slash만 제거한 값

    혹시 팀원 코드에서 진짜 objectKey를 videos/abc.webm 형태로 저장한 경우도 대비한다.
    """
    normalized_key = normalize_video_url_to_object_key(video_url)

    original_key = video_url.strip()
    if original_key.startswith("http://") or original_key.startswith("https://"):
        parsed = urlparse(original_key)
        original_key = parsed.path.lstrip("/")
    else:
        original_key = original_key.lstrip("/")

    candidates = [normalized_key]

    if original_key not in candidates:
        candidates.append(original_key)

    return candidates


def download_video_from_minio(video_url: str) -> str:
    bucket_name = get_bucket_name()

    client = Minio(
        settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=settings.MINIO_SECURE,
    )

    temp_dir = Path(gettempdir()) / "interviewfit-videos"
    temp_dir.mkdir(parents=True, exist_ok=True)

    last_error = None

    for object_key in build_candidate_object_keys(video_url):
        local_path = temp_dir / Path(object_key).name

        try:
            client.fget_object(
                bucket_name,
                object_key,
                str(local_path),
            )

            return str(local_path)

        except S3Error as e:
            last_error = e

            # object key 후보가 틀린 경우 다음 후보로 재시도
            if e.code == "NoSuchKey":
                continue

            raise

    raise RuntimeError(
        f"MinIO에서 영상을 찾지 못했습니다. "
        f"bucket={bucket_name}, video_url={video_url}, last_error={last_error}"
    )


def resolve_video_path(video_url: str) -> str:
    local_path = Path(video_url)

    if local_path.exists():
        return str(local_path)

    return download_video_from_minio(video_url)