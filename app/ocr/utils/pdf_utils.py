import io

from PIL import Image
from pdf2image import convert_from_bytes
from fastapi import UploadFile

async def read_bytes(file: UploadFile) -> bytes:
    return await file.read()

def is_pdf(file: UploadFile) -> bool:
    return file.content_type == "application/pdf"

def to_images(file_bytes: bytes):
    return convert_from_bytes(file_bytes)

def to_image(file_bytes: bytes):
    return Image.open(io.BytesIO(file_bytes))