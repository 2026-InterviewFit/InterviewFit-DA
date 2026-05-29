from app.frame.utils.frame_extractor import extract_frames


def analyze_frames(video_path: str):

    frames = extract_frames(video_path)

    print(f"추출된 프레임 수: {len(frames)}")

    return {
        "frameCount": len(frames)
    }