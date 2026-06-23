from fastapi import APIRouter

from app.VideoStorage.video_router import video_router
from app.eye.eye_blinking.apis.eye_router import router as eye_router
from app.followup.apis.followup_router import router as followup_router
from app.pose.apis.pose_router import router as pose_router
from app.question.apis.question_router import router as question_router
from app.speech_coaching.api.speech_router import router as speech_router

api_router = APIRouter()

# eye_router 자체에는 prefix가 없어서 여기서 /api/eye-blinking을 붙인다.
api_router.include_router(
    eye_router,
    prefix="/api/eye-blinking",
    tags=["EyeBlinking"],
)

# followup_router는 내부에 prefix="/api/followup"이 이미 있음.
api_router.include_router(followup_router)

# question_router는 내부에 prefix="/api/question"이 이미 있음.
api_router.include_router(question_router)

# pose_router는 내부에 prefix="/pose"가 이미 있음.
api_router.include_router(pose_router)

# speech_router는 내부 prefix가 없어서 여기서 /speech를 붙인다.
api_router.include_router(
    speech_router,
    prefix="/speech",
    tags=["Speech"],
)

# video_router는 내부에 prefix="/VideoStorage"가 이미 있음.
api_router.include_router(video_router)