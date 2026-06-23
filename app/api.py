from fastapi import APIRouter

from app.eye_blinking.router import router as eye_router
from app.followup.router import router as followup_router
from app.question.router import router as question_router
from app.pose.router import router as pose_router
from app.speech_coaching.router import router as speech_router
from app.videos.router import router as videos_router

api_router = APIRouter(prefix="/api")

api_router.include_router(eye_router, prefix="/eye-blinking", tags=["EyeBlinking"])
api_router.include_router(followup_router, prefix="/followup", tags=["FollowUp"])
api_router.include_router(question_router, prefix="/question", tags=["Question"])
api_router.include_router(pose_router, prefix="/pose", tags=["Pose"])
api_router.include_router(speech_router, prefix="/speech", tags=["Speech"])
api_router.include_router(videos_router, prefix="/videos", tags=["Videos"])