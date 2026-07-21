from fastapi import APIRouter

from app.embedding.schemas.embedding_schema import CoverLetterEmbeddingRequest, CoverLetterEmbeddingResponse
from app.embedding.services.embedding_service import create_embedding

router = APIRouter()

@router.post("/coverletter", response_model=CoverLetterEmbeddingResponse)
async def embedding(request: CoverLetterEmbeddingRequest):
    return await create_embedding(contents=request.contents)