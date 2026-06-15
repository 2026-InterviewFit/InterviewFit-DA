import logging
import os
from pathlib import Path
from tempfile import gettempdir
from urllib.parse import urlparse

import boto3

logger = logging.getLogger(__name__)


def get_minio_endpoint_for_boto3() -> str:
    """
    boto3는 endpoint_url에 http:// 또는 https://가 필요하다.

    .env의 MINIO_ENDPOINT는 eye 쪽 minio 라이브러리와 같이 쓰기 위해
    localhost:9000 형태로 유지하고, boto3에서만 http://를 붙인다.
    """
    endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000").strip()

    if not endpoint.startswith("http://") and not endpoint.startswith("https://"):
        endpoint = "http://" + endpoint

    return endpoint


def get_minio_access_key() -> str:
    return os.getenv("MINIO_ACCESS_KEY", "minioadmin")


def get_minio_secret_key() -> str:
    return os.getenv("MINIO_SECRET_KEY", "minioadmin123")


def get_minio_bucket_name() -> str:
    return os.getenv("MINIO_BUCKET", "videos")


def get_minio_client():
    return boto3.client(
        "s3",
        endpoint_url=get_minio_endpoint_for_boto3(),
        aws_access_key_id=get_minio_access_key(),
        aws_secret_access_key=get_minio_secret_key(),
    )


def normalize_object_key(source_value: str) -> str:
    if source_value is None or source_value.strip() == "":
        raise ValueError("source_value가 비어 있습니다.")

    value = source_value.strip()

    if value.startswith("http://") or value.startswith("https://"):
        parsed = urlparse(value)
        value = parsed.path.lstrip("/")

    return value.lstrip("/")


def build_candidate_object_keys(source_value: str) -> list[str]:
    """
    Spring VideoService는 현재 objectKey를 보통 이렇게 저장한다.

    bucket = videos
    objectKey = videos/uuid_sample.mp4

    따라서 source_value가 videos/uuid_sample.mp4이면
    그 값을 우선 그대로 시도해야 한다.
    """
    bucket_name = get_minio_bucket_name()
    object_key = normalize_object_key(source_value)

    candidates = []

    if object_key not in candidates:
        candidates.append(object_key)

    bucket_prefix = bucket_name + "/"

    if object_key.startswith(bucket_prefix):
        without_bucket = object_key[len(bucket_prefix):]
        if without_bucket not in candidates:
            candidates.append(without_bucket)

    with_bucket = bucket_prefix + object_key

    if with_bucket not in candidates:
        candidates.append(with_bucket)

    return candidates


def download_video_from_minio(object_name: str, local_path: str | None = None) -> str:
    s3 = get_minio_client()
    bucket_name = get_minio_bucket_name()

    temp_dir = Path(gettempdir()) / "interviewfit-speech-videos"
    temp_dir.mkdir(parents=True, exist_ok=True)

    last_error = None

    for object_key in build_candidate_object_keys(object_name):
        target_path = local_path

        if target_path is None:
            target_path = str(temp_dir / Path(object_key).name)

        try:
            s3.download_file(bucket_name, object_key, target_path)
            logger.info("MinIO 다운로드 성공: bucket=%s, objectKey=%s", bucket_name, object_key)
            return target_path

        except Exception as e:
            logger.warning("MinIO 다운로드 후보 실패: bucket=%s, objectKey=%s, error=%s", bucket_name, object_key, e)
            last_error = e

    raise RuntimeError(
        f"MinIO 영상 다운로드 실패: "
        f"bucket={bucket_name}, object_name={object_name}, last_error={last_error}"
    )