from pydantic import BaseModel

class ErrorResponse(BaseModel):
    status: int
    code: str
    message: str