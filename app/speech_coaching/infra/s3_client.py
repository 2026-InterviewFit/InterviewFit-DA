import boto3
import os
from dotenv import load_dotenv

load_dotenv()

def get_minio_client():
    return boto3.client(
        's3',
        endpoint_url=os.getenv("MINIO_ENDPOINT"), # 예: http://localhost:9000
        aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("MINIO_SECRET_KEY")
    )

def download_video_from_minio(object_name, local_path):
    s3 = get_minio_client()
    bucket_name = os.getenv("MINIO_BUCKET_NAME") # 예: interview-videos
    s3.download_file(bucket_name, object_name, local_path)
    return local_path