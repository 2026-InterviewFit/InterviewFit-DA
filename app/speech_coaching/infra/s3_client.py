import boto3
import os
import logging

logger = logging.getLogger(__name__)

def get_minio_client():
    return boto3.client(
        's3',
        endpoint_url=os.getenv("MINIO_ENDPOINT"),
        aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("MINIO_SECRET_KEY")
    )

def download_video_from_minio(object_name, local_path):
    s3 = get_minio_client()
    bucket_name = os.getenv("MINIO_BUCKET")
    print(f"DEBUG: 현재 로드된 bucket_name: {bucket_name}")
    try:
        s3.download_file(bucket_name, object_name, local_path)
        logger.info(f"MinIO 다운로드 성공: {object_name}")
    except Exception as e:
        logger.error(f"MinIO 다운로드 실패: {e}")
        raise e
    bucket_name = os.getenv("MINIO_BUCKET") # 예: interview-videos
    s3.download_file(bucket_name, object_name, local_path)
    return local_path

