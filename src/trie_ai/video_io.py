from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import cv2

@dataclass
class FrameSample:
    index: int
    timestamp_s: float
    path: Path

def get_video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()
    return float(fps) if fps > 0 else 30.0

def sample_frames_uniform(video_path: str, out_dir: str, num_frames: int = 8) -> list[FrameSample]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if total <= 0:
        # fallback: sample first N seconds
        total = int(fps * 10)

    # indices evenly spaced in [0, total-1]
    if num_frames <= 1:
        indices = [0]
    else:
        indices = [round(i * (total - 1) / (num_frames - 1)) for i in range(num_frames)]

    samples: list[FrameSample] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        ts = float(idx / fps) if fps else 0.0
        fp = out / f"frame_{idx:06d}.jpg"
        cv2.imwrite(str(fp), frame)
        samples.append(FrameSample(index=idx, timestamp_s=ts, path=fp))

    cap.release()
    return samples
