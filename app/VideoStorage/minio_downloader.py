from minio import Minio
import os
import tempfile

minio_client = Minio(
    "localhost:9000",
    access_key="minioadmin",
    secret_key="minioadmin123",
    secure=False
)

BUCKET_NAME = "videos"


def download_video_from_minio(object_key: str) -> str:
    temp_dir = tempfile.gettempdir()
    file_name = os.path.basename(object_key)
    local_path = os.path.join(temp_dir, file_name)

    minio_client.fget_object(
        BUCKET_NAME,
        object_key,
        local_path
    )

    print("MinIO 다운로드 완료:", local_path)

    return local_path