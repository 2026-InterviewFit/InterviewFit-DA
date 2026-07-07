from pydantic import BaseModel

class OcrResponse(BaseModel):
    text: str