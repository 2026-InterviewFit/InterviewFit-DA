from fastapi import APIRouter

from app.speech_coaching.apis.speech_router import router as speech_router
from app.question.apis.question_router import router as question_router
from app.pose.apis.pose_router import router as pose_router
from app.followup.apis.followup_router import router as followup_router
from app.video_storage.video_router import video_router

api_router = APIRouter(prefix="/apis")

api_router.include_router(
    speech_router,
    prefix="/speech",
    tags=["Speech"],
)

api_router.include_router(
    pose_router,
    prefix="/pose",
    tags=["Pose"],
)

api_router.include_router(
    question_router,
    prefix="/question",
    tags=["Question"],
)

api_router.include_router(
    followup_router,
    prefix="/followup",
    tags=["FollowUp"],
)

api_router.include_router(
    video_router,
    prefix="/videos",
    tags=["Videos"],
)