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
    Spring Boot에서 전달하는 영상 경로 기반 분석 요청 DTO.
    camelCase(videoUrl, userId, interviewId)와 snake_case를 모두 허용한다.
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

    # 디버깅용 필드
    min_ear: float | None = None
    max_ear: float | None = None
    ear_threshold: float | None = None
    ear_consec_frames: int | None = None
    closed_frame_count: int | None = None

    # Spring DB 저장용 필드
    blink_rate: float
    score: float
    feedback: str