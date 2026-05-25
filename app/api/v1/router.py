from fastapi import APIRouter

from app.api.v1.endpoints import followup, vision

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(vision.router)
api_router.include_router(followup.router)