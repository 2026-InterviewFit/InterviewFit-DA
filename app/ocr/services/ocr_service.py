import pytesseract
from fastapi import UploadFile

from app.common.exceptions.custom_exception import CustomException
from app.common.exceptions.error_code import ErrorCode
from app.common.exceptions.status_code import StatusCode
from app.ocr.utils.pdf_utils import read_bytes, is_pdf, to_images

async def extract_text(file: UploadFile) -> str | None:
    try:
        file_bytes = await read_bytes(file)

        if is_pdf(file):
            images = to_images(file_bytes)
            if not images:
                raise CustomException(
                    status=StatusCode.BAD_REQUEST,
                    code=ErrorCode.PDF_IMAGE_EXTRACTION_FAILED,
                    message="PDF에서 이미지를 추출할 수 없습니다."
                )

            texts = []

            for idx, img in enumerate(images):
                page_text = pytesseract.image_to_string(img, lang="kor+eng")
                texts.append(page_text)

            text = "\n".join(texts).strip()

            if not text:
                raise CustomException(
                    status=StatusCode.BAD_REQUEST,
                    code=ErrorCode.OCR_FAILED,
                    message="PDF에서 텍스트를 추출할 수 없습니다."
                )
            return text
    except CustomException:
        raise
    except Exception:
        raise CustomException(
            status=StatusCode.INTERNAL_SERVER_ERROR,
            code=ErrorCode.OCR_SERVER_ERROR,
            message="OCR 처리 중 알 수 없는 오류가 발생했습니다."
        )