"""End-to-end: a YouTube URL or a local video file -> match recap.

Orchestrates download (URL only) -> asr -> fuse -> recap as subprocesses, so each
model frees GPU memory before the next. Used by app.py and runnable on its own:

    python src/pipeline.py "https://youtu.be/..."
    python src/pipeline.py data/raw/mymatch.mp4
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

import config

SRC = Path(__file__).resolve().parent
PY = sys.executable

# Score in a title, e.g. "Barcelona 3 vs 3 Inter", "Man Utd 4-4 Everton".
TITLE_SCORE = re.compile(r"(\d{1,2})\s*(?:-|vs\.?|v|x|:)\s*(\d{1,2})", re.I)


def is_url(s):
    return str(s).startswith(("http://", "https://"))


def _run(script, *args):
    subprocess.run([PY, str(SRC / script), *map(str, args)], check=True)


def video_meta(source):
    """Return (duration_seconds, title). URL -> yt-dlp, local file -> ffprobe."""
    if is_url(source):
        r = subprocess.run(
            [PY, "-m", "yt_dlp", "--skip-download", "--print", "%(duration)s|%(title)s", source],
            capture_output=True, text=True,
        )
        line = [l for l in r.stdout.splitlines() if "|" in l]
        dur, _, title = (line[-1] if line else "0|").partition("|")
        return float(dur or 0), title
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nk=1:nw=1", str(source)],
        capture_output=True, text=True,
    )
    return float(r.stdout.strip() or 0), Path(source).stem


def title_score(title):
    m = TITLE_SCORE.search(title or "")
    if m and int(m.group(1)) <= 20 and int(m.group(2)) <= 20:
        return f"{int(m.group(1))}-{int(m.group(2))}"
    return None


def process(source, stem, progress=lambda s: None, whisper="medium", final=None, title=None):
    """Run the full pipeline; return the path to the recap markdown.

    `final`/`title` are trusted hints (e.g. from an upload filename). For a URL they
    default to the video title; the title is passed to the recap so it can name the
    teams, and a score in it anchors the final result.
    """
    if is_url(source):
        progress("Downloading video")
        if title is None:
            title = video_meta(source)[1]
        _run("download.py", source, "--name", stem)
        video = config.RAW / f"{stem}.mp4"
    else:
        video = Path(source)
        if title is None:
            title = video.stem
    if final is None:
        final = title_score(title)

    progress("Transcribing commentary")
    _run("asr.py", video, "--model", whisper)
    progress("Reading scoreboard + audio")
    _run("fuse.py", video)
    progress("Writing recap")
    args = ["recap.py", video, "--title", title]
    if final:
        args += ["--final", final]
    _run(*args)
    progress("Done")
    return config.out_dir(video) / "recap.md"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", help="YouTube URL or local video path")
    p.add_argument("--name", default="cli_job", help="output name (URL only)")
    p.add_argument("--whisper", default="medium")
    a = p.parse_args()
    out = process(a.source, a.name, progress=lambda s: print(f"[{s}]"), whisper=a.whisper)
    print(f"\nRecap: {out}")
