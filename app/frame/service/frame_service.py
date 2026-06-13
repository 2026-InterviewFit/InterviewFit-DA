from app.frame.utils.frame_extractor import extract_frames


def analyze_frames(video_path: str):
    result = extract_frames(video_path)

    print(f"추출된 프레임 수: {result['frameCount']}")
    print(f"프레임 저장 위치: {result['outputDir']}")

    return {
        "frameCount": result["frameCount"],
        "framePaths": result["framePaths"],
        "outputDir": result["outputDir"]
    }
