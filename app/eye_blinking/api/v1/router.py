from fastapi import APIRouter

from app.eye_blinking.api.v1.endpoints import vision

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(vision.router)