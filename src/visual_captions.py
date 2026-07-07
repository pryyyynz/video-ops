"""Timed visual descriptions: detect scene changes, caption a keyframe per scene.

ffmpeg's scene-change filter finds the cuts; one keyframe per scene goes to a
local vision LLM (gemma3 via Ollama, already pulled for the recap) for a
one-sentence description. Writes out/<video>/visual_captions.json with the same
{start, end, text} segment shape as the transcript, so the app renders and
exports both identically. Also the visual evidence for summarise.py on
speech-poor videos.

Usage:
    python src/visual_captions.py data/raw/video.mp4 --max-scenes 30
"""
import argparse
import base64
import json
import math
import re
import subprocess

import cv2

import config
import llm

MODEL = "gemma3:12b"           # vision-capable, already pulled for the recap step
SCENE_THRESHOLD = 0.30
MIN_GAP = 8.0                  # seconds between kept scene boundaries
FALLBACK_STEP = 30.0           # static videos (no cuts): sample on a fixed grid
MAX_WIDTH = 896                # downscale frames before sending to the model
PROMPT = ("Describe what is happening in this video frame in one factual sentence: "
          "the visible setting and action. No preamble, no speculation.")


def mmss(seconds):
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


def duration_of(video):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nk=1:nw=1", str(video)],
        capture_output=True, text=True,
    )
    return float(r.stdout.strip() or 0)


def detect_scenes(video, duration, max_scenes):
    """Scene-boundary times, thinned to MIN_GAP and capped at max_scenes."""
    r = subprocess.run(
        ["ffmpeg", "-i", str(video), "-vf",
         f"select='gt(scene,{SCENE_THRESHOLD})',showinfo", "-an", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    bounds = [0.0]
    for t in (float(x) for x in re.findall(r"pts_time:([0-9.]+)", r.stderr)):
        if t - bounds[-1] >= MIN_GAP:
            bounds.append(t)
    if len(bounds) < 3:  # barely any cuts (static camera / slides): fixed grid
        bounds = [i * FALLBACK_STEP for i in range(max(1, int(duration // FALLBACK_STEP) + 1))]
    if len(bounds) > max_scenes:
        step = math.ceil(len(bounds) / max_scenes)
        bounds = bounds[::step]
    return bounds


def frame_b64(cap, t):
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    if not ok:
        return None
    h, w = frame.shape[:2]
    if w > MAX_WIDTH:
        frame = cv2.resize(frame, (MAX_WIDTH, int(h * MAX_WIDTH / w)))
    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(jpg.tobytes()).decode() if ok else None


def run(video_path, max_scenes=30, model=MODEL):
    duration = duration_of(video_path)
    bounds = detect_scenes(video_path, duration, max_scenes)
    ends = bounds[1:] + [duration]
    print(f"{len(bounds)} scene(s) over {duration:.0f}s; captioning with {model}...")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    segments = []
    for a, b in zip(bounds, ends):
        img = frame_b64(cap, min(a + 1.0, (a + b) / 2))
        if img is None:
            continue
        text = " ".join(llm.ask(PROMPT, model, num_ctx=4096, images=[img], temperature=0.2).split())
        segments.append({"start": round(a, 1), "end": round(b, 1), "text": text})
        print(f"  [{mmss(a)}] {text}")
    cap.release()

    out = config.out_dir(video_path) / "visual_captions.json"
    out.write_text(json.dumps({"video": str(video_path), "segments": segments}, indent=2),
                   encoding="utf-8")
    print(f"\n{len(segments)} caption(s) -> {out}")
    return segments


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--max-scenes", type=int, default=30)
    p.add_argument("--model", default=MODEL)
    a = p.parse_args()
    run(a.video, a.max_scenes, a.model)
