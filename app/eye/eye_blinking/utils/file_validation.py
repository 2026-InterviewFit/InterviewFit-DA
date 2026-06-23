from fastapi import HTTPException, UploadFile

ALLOWED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
ALLOWED_VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".webm", ".mkv")


def validate_upload_file(
    upload_file: UploadFile,
    allowed_extensions: tuple[str, ...],
    expected_type_prefix: str,
    type_name: str,
) -> None:
    """
    업로드 파일이 허용된 확장자 또는 Content-Type을 가지는지 검증한다.
    둘 중 하나만 맞아도 통과한다.
    """
    filename_ok = (
        upload_file.filename is not None
        and upload_file.filename.lower().endswith(allowed_extensions)
    )

    content_type_ok = (
        upload_file.content_type is not None
        and upload_file.content_type.startswith(expected_type_prefix)
    )

    if not filename_ok and not content_type_ok:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{type_name} 파일만 업로드 가능합니다. "
                f"(content_type: {upload_file.content_type}, "
                f"filename: {upload_file.filename})"
            ),
        )