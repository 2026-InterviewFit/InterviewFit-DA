import httpx

from app.core.config import settings
from app.embedding.schemas.embedding_schema import CoverLetterEmbeddingResponse

async def create_embedding(contents: list[str]) -> CoverLetterEmbeddingResponse:
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{settings.TEI_URL}/embed",
            json={
                "inputs": contents
            }
        )

        response.raise_for_status()
        embeddings = response.json()

        return CoverLetterEmbeddingResponse(
            embeddings=embeddings
        )