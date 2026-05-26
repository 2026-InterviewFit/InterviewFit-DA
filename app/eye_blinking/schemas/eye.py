from typing import List

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class EyeLandmark(BaseModel):
    index: int
    x: float
    y: float
    z: float
    pixel_x: int
    pixel_y: int


class EyeLandmarksResponse(BaseModel):
    face_detected: bool
    image_width: int
    image_height: int
    left_eye_landmarks: List[EyeLandmark]
    right_eye_landmarks: List[EyeLandmark]


class EyeBlinkingAnalyzeRequest(BaseModel):
    """
    Spring Boot에서 전달하는 영상 경로 기반 분석 요청 DTO

    Spring 요청 예시:
    {
      "videoUrl": "/Users/.../uploads/videos/test.mp4",
      "userId": 1,
      "interviewId": 1
    }
    """

    model_config = ConfigDict(
        populate_by_name=True,
        extra="ignore",
    )

    video_url: str = Field(
        ...,
        validation_alias=AliasChoices("video_url", "videoUrl"),
    )

    user_id: int | None = Field(
        default=None,
        validation_alias=AliasChoices("user_id", "userId"),
    )

    interview_id: int | None = Field(
        default=None,
        validation_alias=AliasChoices("interview_id", "interviewId"),
    )


class BlinkAnalysisResponse(BaseModel):
    total_frames: int
    fps: float
    duration_seconds: float
    face_detected_frames: int

    blink_count: int
    blinks_per_minute: float
    average_ear: float
    blink_timestamps: List[float]
    evaluation: str

    # Spring DB 저장용 필드
    blink_rate: float
    score: float
    feedback: str