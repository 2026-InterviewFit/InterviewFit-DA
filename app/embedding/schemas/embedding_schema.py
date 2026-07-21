from pydantic import BaseModel

class CoverLetterEmbeddingRequest(BaseModel):
    contents: list[str]

class CoverLetterEmbeddingResponse(BaseModel):
    embeddings: list[list[float]]