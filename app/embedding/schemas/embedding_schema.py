from pydantic import BaseModel

class CoverLetterEmbeddingRequest(BaseModel):
    text: str

class CoverLetterEmbeddingResponse(BaseModel):
    embedding: list[float]