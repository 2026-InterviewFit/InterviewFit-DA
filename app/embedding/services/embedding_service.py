import httpx

from app.core.config import settings

async def create_embedding(text: str) -> list[float]:
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{settings.TEI_URL}/embed",
            json={
                "inputs": text
            }
        )
        response.raise_for_status()
        result = response.json()
        return result[0]