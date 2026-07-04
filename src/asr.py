"""Transcribe the commentary track with Whisper (timestamped segments).

Commentary is the richest "what happened" signal in a football broadcast - goals,
names, near-misses, cards. Writes data/transcript.json for the recap step.

Usage:
    python src/asr.py data/raw/clip.mp4 --model small
"""
import argparse
import json

import torch
import whisper

import config


def mmss(seconds):
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


def run(video_path, model_name="medium"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Whisper '{model_name}' on {device}...")
    model = whisper.load_model(model_name, device=device)

    print("Transcribing the whole audio track...")
    result = model.transcribe(str(video_path), fp16=(device == "cuda"))  # auto-detect language
    segments = [
        {"start": round(s["start"], 1), "end": round(s["end"], 1), "text": s["text"].strip()}
        for s in result["segments"]
    ]

    out = config.out_dir(video_path) / "transcript.json"
    out.write_text(
        json.dumps({"video": str(video_path), "segments": segments}, indent=2),
        encoding="utf-8",
    )
    print(f"\n{len(segments)} segments -> {out}")
    for s in segments[:8]:
        print(f"  [{mmss(s['start'])}] {s['text']}")
    return segments


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--model", default="medium", help="tiny|base|small|medium|large-v3")
    a = p.parse_args()
    run(a.video, a.model)
