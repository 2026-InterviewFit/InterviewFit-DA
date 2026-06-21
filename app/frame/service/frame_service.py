import logging

from app.frame.utils.frame_extractor import extract_frames

logger = logging.getLogger(__name__)


def analyze_frames(video_path: str):
    result = extract_frames(video_path)

    logger.info("추출된 프레임 수: %s", result["frameCount"])
    logger.info("프레임 저장 위치: %s", result["outputDir"])

    return {
        "frameCount": result["frameCount"],
        "framePaths": result["framePaths"],
        "outputDir": result["outputDir"]
    }
