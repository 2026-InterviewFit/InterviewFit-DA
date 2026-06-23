from fastapi import APIRouter

from app.VideoStorage.video_router import video_router
from app.eye.eye_blinking.apis.eye_router import router as eye_router
from app.followup.apis.followup_router import router as followup_router
from app.pose.apis.pose_router import router as pose_router
from app.question.apis.question_router import router as question_router
from app.speech_coaching.apis.speech_router import router as speech_router

api_router = APIRouter()

api_router.include_router(
    eye_router,
    prefix="/api/eye-blinking",
    tags=["EyeBlinking"],
)

api_router.include_router(followup_router)
api_router.include_router(question_router)
api_router.include_router(pose_router)

api_router.include_router(
    speech_router,
    prefix="/speech",
    tags=["Speech"],
)

api_router.include_router(video_router)
