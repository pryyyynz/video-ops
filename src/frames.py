"""Extract sample frames from a video for calibration / debugging.

Usage:
    python src/frames.py data/raw/clip.mp4 --count 12
    python src/frames.py data/raw/clip.mp4 --fps 1
"""
import argparse
from pathlib import Path

import cv2

import config


def extract(video_path, out_dir, count=None, fps=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration = total / src_fps

    if count:
        targets = [duration * i / (count + 1) for i in range(1, count + 1)]
    else:
        step = 1.0 / (fps or config.SAMPLE_FPS)
        targets = [i * step for i in range(int(duration / step))]

    saved = []
    for t in targets:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        name = out_dir / f"frame_{t:07.2f}s.png"
        cv2.imwrite(str(name), frame)
        saved.append(name)

    cap.release()
    print(f"Saved {len(saved)} frames to {out_dir}")
    return saved


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--count", type=int, default=12, help="N evenly-spaced frames")
    p.add_argument("--fps", type=float, help="sample at this fps instead of --count")
    p.add_argument("--out", default=str(config.FRAMES))
    a = p.parse_args()
    extract(a.video, a.out, count=None if a.fps else a.count, fps=a.fps)
