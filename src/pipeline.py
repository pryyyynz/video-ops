"""End-to-end: a YouTube URL or a local video file -> selected task outputs.

Orchestrates the steps as subprocesses, so each model frees GPU memory before
the next. Shared steps (download, ASR, fuse, scene captions) run once per job
no matter how many selected tasks need them; then each task's finisher runs in
order. Used by app.py and runnable on its own:

    python src/pipeline.py "https://youtu.be/..." --tasks football_recap
    python src/pipeline.py data/raw/video.mp4 --tasks subtitles,highlights
"""
import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import config

SRC = Path(__file__).resolve().parent


def _step_python():
    """The step scripts need the video_summ env (whisper/easyocr/torch), but app.py
    may be launched from base or another env. Prefer the current interpreter when it
    has the deps; otherwise locate the env's python instead of failing in the steps."""
    if importlib.util.find_spec("whisper"):
        return sys.executable
    prefix = Path(sys.prefix)
    for env in (prefix / "envs" / "video_summ",          # launched from conda base
                prefix.parent / "video_summ",            # launched from a sibling env
                Path.home() / "miniforge3" / "envs" / "video_summ"):
        exe = env / ("python.exe" if os.name == "nt" else "bin/python")
        if exe.exists():
            print(f"[pipeline] this python lacks the step deps; using {exe}")
            return str(exe)
    return sys.executable  # last resort: steps will fail with a clear import error


PY = _step_python()

# Score in a title, e.g. "Barcelona 3 vs 3 Inter", "Man Utd 4-4 Everton".
TITLE_SCORE = re.compile(r"(\d{1,2})\s*(?:-|vs\.?|v|x|:)\s*(\d{1,2})", re.I)

# Cheap-first run order; also which shared steps each task depends on.
RUN_ORDER = ["subtitles", "visual_captions", "highlights", "video_summary", "football_recap"]
NEEDS_ASR = {"subtitles", "highlights", "video_summary", "football_recap"}
NEEDS_FUSE = {"highlights", "football_recap"}

SPARSE_WORDS = 200  # below this, a summary can't lean on speech alone


def is_url(s):
    return str(s).startswith(("http://", "https://"))


def _run(script, *args):
    r = subprocess.run([PY, str(SRC / script), *map(str, args)],
                       stderr=subprocess.PIPE, text=True, errors="replace")
    if r.stderr:
        sys.stderr.write(r.stderr)  # keep step logs visible on the console
    if r.returncode:
        tail = "\n".join((r.stderr or "").strip().splitlines()[-6:])
        raise RuntimeError(f"{script} failed (exit {r.returncode}):\n{tail}")


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


def transcript_words(video):
    try:
        segments = json.loads(
            (config.out_dir(video) / "transcript.json").read_text(encoding="utf-8"))["segments"]
    except FileNotFoundError:
        return 0
    return sum(len(s["text"].split()) for s in segments)


def run_tasks(source, stem, tasks, *, whisper="medium", final=None, title=None, target=300,
              progress=lambda s: None, task_start=lambda t: None, task_done=lambda t, v: None):
    """Run the selected tasks on one input; returns the local video path.

    `task_done(task, video)` fires as each task's artifact lands, so callers can
    surface results incrementally. `final`/`title` are trusted hints for the
    football recap (from the upload filename or URL title).
    """
    tasks = [t for t in RUN_ORDER if t in tasks]

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

    if set(tasks) & NEEDS_ASR:
        progress("Transcribing audio")
        _run("asr.py", video, "--model", whisper)
    if set(tasks) & NEEDS_FUSE:
        progress("Reading scoreboard + audio")
        _run("fuse.py", video)

    # Scene captions: their own task, and the fallback evidence for a summary
    # of a video with too little speech.
    need_visual = "visual_captions" in tasks
    if "video_summary" in tasks and not need_visual and transcript_words(video) < SPARSE_WORDS:
        need_visual = True
    if need_visual:
        progress("Describing scenes")
        _run("visual_captions.py", video)

    for task in tasks:
        task_start(task)
        if task == "football_recap":
            progress("Writing recap")
            args = ["recap.py", video, "--title", title]
            if final:
                args += ["--final", final]
            _run(*args)
        elif task == "video_summary":
            progress("Writing summary")
            _run("summarise.py", video, "--title", title)
        elif task == "highlights":
            progress("Cutting highlights")
            _run("highlights.py", video, "--target", target)
        # subtitles / visual_captions: the shared steps already made their artifacts
        task_done(task, video)

    progress("Done")
    return video


def process(source, stem, progress=lambda s: None, whisper="medium", final=None, title=None):
    """Back-compat football-recap-only entry point; returns the recap path."""
    video = run_tasks(source, stem, ["football_recap"], whisper=whisper,
                      final=final, title=title, progress=progress)
    return config.out_dir(video) / "recap.md"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", help="YouTube URL or local video path")
    p.add_argument("--tasks", default="football_recap",
                   help=f"comma-separated: {','.join(RUN_ORDER)}")
    p.add_argument("--name", default="cli_job", help="output name (URL only)")
    p.add_argument("--whisper", default="medium")
    p.add_argument("--target", type=int, default=300, help="highlights length, seconds")
    a = p.parse_args()
    video = run_tasks(a.source, a.name, a.tasks.split(","), whisper=a.whisper, target=a.target,
                      progress=lambda s: print(f"[{s}]"),
                      task_done=lambda t, v: print(f"[{t}: done]"))
    print(f"\nOutputs in: {config.out_dir(video)}")
