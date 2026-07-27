import httpx

from app.core.config import settings
from app.embedding.schemas.embedding_schema import CoverLetterEmbeddingResponse

# 나중에 lifespan으로 빼기 (또는 clients 파일로 빼기)
tei_client = httpx.AsyncClient(timeout=60.0)

async def create_embedding(contents: list[str]) -> CoverLetterEmbeddingResponse:
    response = await tei_client.post(
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