from app.frame.utils.frame_extractor import extract_frames


def analyze_frames(video_path: str) -> dict:
    frame_result = extract_frames(video_path)

    return {
        "frameCount": frame_result["frameCount"],
        "framePaths": frame_result["framePaths"],
        "outputDir": frame_result["outputDir"],
    }
