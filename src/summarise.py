"""Generic video summary from the transcript + optional visual timeline.

Speech-rich videos (talks, tutorials, vlogs) are summarised from the Whisper
transcript. When speech is thin, the pipeline first runs visual_captions.py and
this step leans on the scene descriptions instead. Writes out/<video>/summary.md.

Usage:
    python src/summarise.py data/raw/video.mp4 --title "How to ..."
"""
import argparse
import json

import config
import llm

MODEL = "gemma3:12b"
NUM_CTX = 16384

PROMPT = """Summarise this video in ENGLISH from the evidence below. The transcript may be
in another language; translate as needed. Ground every claim in the evidence and do not
invent specifics.{title_line}

TRANSCRIPT (MM:SS video time){transcript_note}:
{transcript}

VISUAL TIMELINE (scene descriptions with MM:SS):
{visual}

Write markdown, plain text, no emoji, in two sections:

## Summary
5-8 sentences: what kind of video this is, what happens or is argued, and the key
points or turning moments, in order.

## Chapters
5-10 bullets, one per distinct part, in time order: "- [MM:SS] short label". Use the
timestamps from the evidence.
"""


def mmss(seconds):
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


def fmt_segments(segments):
    return "\n".join(f"[{mmss(s['start'])}] {s['text']}" for s in segments if s["text"])


def run(video, title=None, model=MODEL):
    d = config.out_dir(video)
    transcript = json.loads((d / "transcript.json").read_text(encoding="utf-8"))["segments"]
    visual_path = d / "visual_captions.json"
    visual = json.loads(visual_path.read_text(encoding="utf-8"))["segments"] if visual_path.exists() else []

    words = sum(len(s["text"].split()) for s in transcript)
    sparse = words < 200
    prompt = PROMPT.format(
        title_line=f"\nVIDEO TITLE: {title}" if title else "",
        transcript_note=" - very sparse; rely on the visual timeline" if sparse else "",
        transcript=fmt_segments(transcript) or "(no speech detected)",
        visual=fmt_segments(visual) or "(none)",
    )
    print(f"Asking {model} ({words} transcript words, {len(visual)} scene captions)...\n")
    summary = llm.ask(prompt, model, num_ctx=NUM_CTX)
    out = d / "summary.md"
    out.write_text(summary, encoding="utf-8")
    try:
        print(summary)
    except UnicodeEncodeError:
        print(summary.encode("ascii", "replace").decode())
    print(f"\n\nSaved to {out}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--title", help="video title / filename, gives the model context")
    p.add_argument("--model", default=MODEL)
    a = p.parse_args()
    run(a.video, a.title, a.model)
