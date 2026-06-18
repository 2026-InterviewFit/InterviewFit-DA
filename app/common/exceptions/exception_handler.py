from fastapi import Request
from fastapi.responses import JSONResponse

from app.common.exceptions.custom_exception import CustomException
from app.common.exceptions.schemas.error_schema import ErrorResponse


async def custom_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, CustomException)

    error_response = ErrorResponse(
        status=exc.status,
        code=exc.code,
        message=exc.message
    )

    return JSONResponse(
        status_code=exc.status,
        content=error_response.model_dump()
    )